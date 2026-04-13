# bandit_train_batch.py
# Train + evaluate IL/AO/OL bandit agent.
# Includes:
#   - early-stopped training loop
#   - fixed-prob eval + per-trial plot
#   - gap sweep eval: p_hi fixed, p_lo swept
#
# Notes:
#   - This file is designed to be backward-compatible with older var_bandit_learner2.py
#     (no teacher_correct_buf argument). If your learner supports teacher_correct_buf,
#     it will be passed; otherwise it will be ignored.

import argparse
import os
from datetime import datetime
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

import ray

import visual_bandit_env3 as vbe

import gains_model as bl  # type: ignore

from torch.utils.tensorboard import SummaryWriter
import os
from datetime import datetime



# -----------------------------
# Utils
# -----------------------------

def extract_lr_views(obs_tensor: torch.Tensor, env, crop_size: int = 112, pad: int = 6) -> Tuple[torch.Tensor, torch.Tensor]:
    """Crop left/right bandit patches from the full-frame observation and resize to (crop_size,crop_size)."""
    lr = env.unwrapped.left_rect
    rr = env.unwrapped.right_rect
    _, _, H, W = obs_tensor.shape

    y1 = max(min(lr.top, rr.top) - pad, 0)
    y2 = min(max(lr.bottom, rr.bottom) + pad, H)

    def crop_resize(rect):
        x1 = max(rect.left - pad, 0)
        x2 = min(rect.right + pad, W)
        patch = obs_tensor[:, :, y1:y2, x1:x2]
        patch = F.interpolate(patch, size=(crop_size, crop_size), mode="bilinear", align_corners=False)
        return patch

    return crop_resize(lr), crop_resize(rr)


def save_checkpoint(path: str, agent, optim_bandit, optim_motor, extra: Dict[str, Any]) -> None:
    ckpt = {
        "model_state": agent.state_dict(),
        "optim_bandit_state": optim_bandit.state_dict(),
        "optim_motor_state": optim_motor.state_dict(),
        "extra": extra,
        "torch": torch.__version__,
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(ckpt, path)


def sample_probs_this_session(session_K: int, p_hi: float, p_lo: float, rng: np.random.Generator) -> List[Tuple[float, float]]:
    """Randomize which side has higher prob per pair."""
    probs = []
    for _ in range(session_K):
        if rng.random() < 0.5:
            probs.append((p_hi, p_lo))
        else:
            probs.append((p_lo, p_hi))
    return probs


# -----------------------------
# Rollout
# -----------------------------

def meta_ep_rollout(env, agent, device, session_K: int, session_N: int, worker_id: int = 0, print_this_session: bool = False, return_obs: bool = True):
    """One full session rollout producing buffers used by update2.

    Returns a tuple:
      motor buffers: xy_pos_buf, goal_vec_buf, chosen_bandits_motor_buf
      bandit buffers: obs_bandit, chosen_bandits_buf, bandit_rewards_buf, meta_ep_start_buf, actor_buf, trial_cond_buf
      teacher_correct_buf: [-1 for non-observation trials, else 0/1 if teacher chose high reward]
      summary stats: cum_rewards_il/ao/ol, high_reward_choices_{il,ao,ol} (arrays over trials)
    """

    def log(*args, **kwargs):
        if print_this_session and worker_id == 0:
            print(*args, **kwargs)

    # Attention memory (past trials)
    k_tokens = []      # keys: q_in(chosen_feat)
    v_tokens = []      # values: action+actor+reward embeddings
    cond_tokens = []   # trial condition per past trial

    obs, info = env.reset()
    done = False

    pair_index_counter_il = np.ones(session_K, dtype=np.int32) * -1
    pair_index_counter_ao = np.ones(session_K, dtype=np.int32) * -1
    pair_index_counter_ol = np.ones(session_K, dtype=np.int32) * -1
    high_reward_choices_il = np.zeros(session_N, dtype=np.int32)
    high_reward_choice_ao = np.zeros(session_N, dtype=np.int32)
    high_reward_choice_ol = np.zeros(session_N, dtype=np.int32)

    # scalar accuracy counters for self-choice trials
    hi_cnt_il = hi_cnt_ao = hi_cnt_ol = 0
    tot_il    = tot_ao    = tot_ol    = 0

    meta_ep_len = 0
    ep_start_flag = 1.0

    # motor buffers (flat)
    xy_pos_buf: List[np.ndarray] = []
    goal_vec_buf: List[np.ndarray] = []
    chosen_bandits_motor_buf: List[np.ndarray] = []

    # bandit buffers (per trial)
    left_obs = [] if return_obs else None
    right_obs = [] if return_obs else None

    chosen_bandits_buf = []
    bandit_rewards_buf = []
    meta_ep_start_buf = []
    trial_cond_buf = []
    actor_buf = []
    teacher_correct_buf = []

    bandit_action_buf = []
    bandit_logp_buf   = []
    bandit_value_buf  = []
    bandit_policy_mask_buf = []  # 1 for IL/AO-s/OL-s, else 0

    # state for the current trial
    curr_trial_condition = None
    left_feats = None
    right_feats = None
    choice_target = None
    a_t = 0

    p_left = torch.tensor(0.5)
    p_right = torch.tensor(0.5)

    delta_val = np.array([0.0,0.0])
    g_val = 0.0

    def _ctx_for(q_tok: torch.Tensor, keep_conds: set) -> torch.Tensor:
        """Return ctx for a single query token, attending only to past trials in keep_conds.

        q_tok: (1,1,H) ; returns: (1,1,H)
        """
        if len(k_tokens) == 0:
            return agent.attn_ln(q_tok)

        k = torch.stack(k_tokens, dim=0)  # (Lpast,1,H)
        v = torch.stack(v_tokens, dim=0)  # (Lpast,1,H)
        keep = torch.as_tensor([c in keep_conds for c in cond_tokens], device=device, dtype=torch.bool)  # (Lpast,)
        kpm = (~keep).unsqueeze(0)  # (1,Lpast) True=mask
        ctx_attn = agent._safe_causal_attn(q_tok, k, v, kpm)  # (1,1,H)
        return agent.attn_ln(ctx_attn + q_tok)

    def _choice_logits(curr_cond: str, lf: torch.Tensor, rf: torch.Tensor) -> torch.Tensor:
        """Compute (1,1,2) action logits for self-choice trials."""
        ql = agent.q_in(lf).unsqueeze(0)  # (1,1,H)
        qr = agent.q_in(rf).unsqueeze(0)  # (1,1,H)

        lf_seq = lf.unsqueeze(0)  # (1,1,F)
        rf_seq = rf.unsqueeze(0)  # (1,1,F)

        if curr_cond == "IL":
            ctx_l = _ctx_for(ql, {"IL"})
            ctx_r = _ctx_for(qr, {"IL"})
            log_l = agent._head(agent.q_base, ctx_l, lf_seq)
            log_r = agent._head(agent.q_base, ctx_r, rf_seq)
            return torch.cat([log_l, log_r], dim=-1), None, None

        if curr_cond == "AO-s":
            ctx_l_s = _ctx_for(ql, {"AO-s"})
            ctx_r_s = _ctx_for(qr, {"AO-s"})
            base_l = agent._head(agent.q_base, ctx_l_s, lf_seq)
            base_r = agent._head(agent.q_base, ctx_r_s, rf_seq)
            base = torch.cat([base_l, base_r], dim=-1)

            # correction from observed-teacher-action history
            ctx_l_o = _ctx_for(ql, {"AO-o"})
            ctx_r_o = _ctx_for(qr, {"AO-o"})
            d_l = agent._head(agent.q_delta_ao, ctx_l_o, lf_seq)
            d_r = agent._head(agent.q_delta_ao, ctx_r_o, rf_seq)
            delta = torch.cat([d_l, d_r], dim=-1)

            g_in = torch.cat([ctx_l_s, ctx_r_s, ctx_l_o, ctx_r_o], dim=-1)  # (1,1,4H)
            g = torch.sigmoid(agent.gate_ao(g_in)).clamp(0.0, 1.0)         # (1,1,1)
            delta_f = delta * g

            has_prev_ao = any(c in ("AO-o") for c in cond_tokens)
            logit_final = base + delta_f if has_prev_ao else base

            return logit_final, delta, g
        if curr_cond == "OL-s":
            ctx_l_s = _ctx_for(ql, {"OL-s"})
            ctx_r_s = _ctx_for(qr, {"OL-s"})
            base_l = agent._head(agent.q_base, ctx_l_s, lf_seq)
            base_r = agent._head(agent.q_base, ctx_r_s, rf_seq)
            base = torch.cat([base_l, base_r], dim=-1)

            # correction from observed-teacher-feedback history
            ctx_l_o = _ctx_for(ql, {"OL-o"})
            ctx_r_o = _ctx_for(qr, {"OL-o"})
            d_l = agent._head(agent.q_delta_ol, ctx_l_o, lf_seq)
            d_r = agent._head(agent.q_delta_ol, ctx_r_o, rf_seq)
            delta = torch.cat([d_l, d_r], dim=-1)

            g_in = torch.cat([ctx_l_s, ctx_r_s, ctx_l_o, ctx_r_o], dim=-1)  # (1,1,4H)
            g = torch.sigmoid(agent.gate_ol(g_in)).clamp(0.0, 1.0)         # (1,1,1)
            delta = delta * g

            has_prev_ol = any(c in ("OL-o") for c in cond_tokens)
            logit_final = base + delta if has_prev_ol else base
            return logit_final, delta, g

        # Fallback: treat unknown as IL
        ctx_l = _ctx_for(ql, {"IL"})
        ctx_r = _ctx_for(qr, {"IL"})
        log_l = agent._head(agent.q_base, ctx_l, lf_seq)
        log_r = agent._head(agent.q_base, ctx_r, rf_seq)
        return torch.cat([log_l, log_r], dim=-1)
    
    def _value_pred(curr_cond: str, lf: torch.Tensor, rf: torch.Tensor) -> torch.Tensor:
        ql = agent.q_in(lf).unsqueeze(0)  # (1,1,H)
        qr = agent.q_in(rf).unsqueeze(0)

        if curr_cond == "IL":
            ctx_l_s = _ctx_for(ql, {"IL"})
            ctx_r_s = _ctx_for(qr, {"IL"})
            ctx_l_o = torch.zeros_like(ctx_l_s)
            ctx_r_o = torch.zeros_like(ctx_r_s)

        elif curr_cond == "AO-s":
            ctx_l_s = _ctx_for(ql, {"AO-s"})
            ctx_r_s = _ctx_for(qr, {"AO-s"})
            ctx_l_o = _ctx_for(ql, {"AO-o"})
            ctx_r_o = _ctx_for(qr, {"AO-o"})

        elif curr_cond == "OL-s":
            ctx_l_s = _ctx_for(ql, {"OL-s"})
            ctx_r_s = _ctx_for(qr, {"OL-s"})
            ctx_l_o = _ctx_for(ql, {"OL-o"})
            ctx_r_o = _ctx_for(qr, {"OL-o"})

        else:
            # AO-o / OL-o: no policy update, but value can still be trained
            ctx_l_s = _ctx_for(ql, {curr_cond})
            ctx_r_s = _ctx_for(qr, {curr_cond})
            ctx_l_o = torch.zeros_like(ctx_l_s)
            ctx_r_o = torch.zeros_like(ctx_r_s)

        v_in = torch.cat(
            [ctx_l_s, ctx_r_s, ctx_l_o, ctx_r_o, lf.unsqueeze(0), rf.unsqueeze(0)],
            dim=-1
        )  # (1,1,4H+2F)
        v = agent.v_head(v_in)  # (1,1,1)
        return v

    while not done:
        obs_tensor = (
            torch.as_tensor(obs, device=device)
            .permute(2, 0, 1).unsqueeze(0)
            .to(torch.float32).div_(255.0)
        )

        # CHOICE at trial start
        if ep_start_flag == 1.0:
            left_view, right_view = extract_lr_views(obs_tensor, env, crop_size=112, pad=6)
            curr_trial_condition = info.get("curr_trial_condition")

            left_feats = agent.encode(left_view)    # (1,F)
            right_feats = agent.encode(right_view)  # (1,F)

            if return_obs:
                left_obs.append(left_view.squeeze(0).detach().cpu().numpy())
                right_obs.append(right_view.squeeze(0).detach().cpu().numpy())

            # Self-choice trials
            if curr_trial_condition not in ("AO-o", "OL-o"):
                logits, delta, g = _choice_logits(curr_trial_condition, left_feats, right_feats)  # (1,1,2)
                delta_val = np.array(delta[0,0].cpu().numpy()) if delta is not None else np.array([-1,-1])
                g_val = g[0,0].item() if g is not None else -1
                # print("delta.shape = ", delta.shape, ", g.shape = ", g.shape, g_val, delta_val)
                # tau = 0.2
                # p_choose = torch.softmax(logits / tau, dim=-1)  # (1,1,2)
                # p_left = p_choose[0, 0, 0]
                # p_right = p_choose[0, 0, 1]
                # dist = torch.distributions.Categorical(probs=p_choose[0, 0])
                # a_t = int(dist.sample().item())  # 0=left, 1=right


                tau = 1.0  # let entropy bonus drive exploration; you can anneal later
                dist = torch.distributions.Categorical(logits=(logits[0, 0] / tau))
                p_left = torch.softmax(logits[0, 0], dim=-1)[0]
                p_right = torch.softmax(logits[0, 0], dim=-1)[1]
                a_t = int(dist.sample().item())
                logp_old = float(dist.log_prob(torch.tensor(a_t, device=device)).item())
                v_old = float(_value_pred(curr_trial_condition, left_feats, right_feats).item())
                bandit_action_buf.append(a_t)
                bandit_logp_buf.append(logp_old)
                bandit_value_buf.append(v_old)
                bandit_policy_mask_buf.append(1.0)

                choice_target = F.one_hot(torch.tensor(a_t, device=device), num_classes=2).unsqueeze(0).float()
            else:
                # Observational trial: env/teacher chooses (revealed at trial end).
                a_t = 0  # dummy; replaced at trial end
                choice_target = F.one_hot(torch.tensor(a_t, device=device), num_classes=2).unsqueeze(0).float()

                v_old = float(_value_pred(curr_trial_condition, left_feats, right_feats).item())
                bandit_action_buf.append(0)     # dummy
                bandit_logp_buf.append(0.0)     # dummy
                bandit_value_buf.append(v_old)
                bandit_policy_mask_buf.append(0.0)


            meta_ep_len += 1
            ep_start_flag = 0.0

        # MOTOR step
        W, H = env.unwrapped.W, env.unwrapped.H
        x_pix, y_pix = env.unwrapped.cursor
        xy_norm = np.array([(x_pix/(W-1))*2-1, (y_pix/(H-1))*2-1], dtype=np.float32)

        left_c  = np.array([env.unwrapped.left_rect.centerx,  env.unwrapped.left_rect.centery],  np.float32)
        right_c = np.array([env.unwrapped.right_rect.centerx, env.unwrapped.right_rect.centery], np.float32)
        left_c_norm  = np.array([(left_c[0]/(W-1))*2-1,  (left_c[1]/(H-1))*2-1],  np.float32)
        right_c_norm = np.array([(right_c[0]/(W-1))*2-1, (right_c[1]/(H-1))*2-1], np.float32)

        chosen_center = left_c_norm if choice_target.argmax(dim=-1).item() == 0 else right_c_norm
        g_norm = chosen_center - xy_norm

        xy_pos_t   = torch.as_tensor(xy_norm).unsqueeze(0).to(device)
        goal_vec_t = torch.as_tensor(g_norm).unsqueeze(0).to(device)

        mu, log_std = agent.motor_fwd(choice_target.detach(), xy_pos=xy_pos_t, goal_vec=goal_vec_t)

        xy_pos_buf.append(xy_norm)
        goal_vec_buf.append(g_norm)
        chosen_bandits_motor_buf.append(choice_target.squeeze(0).detach().cpu().numpy())

        std = torch.exp(log_std)
        y = mu + std * torch.randn_like(std)
        action = torch.tanh(y)
        action_np = action.squeeze(0).detach().cpu().numpy()

        next_obs, reward, term, _, info = env.step(action_np)
        obs = next_obs
        done = term

        # Trial ended => append memory token
        if info.get("trial_ended"):
            meta_ep_start = 0.0 if meta_ep_len > 1 else 1.0

            actor = 0
            if curr_trial_condition in ("AO-o", "OL-o"):
                # on obs trials, use env-chosen action
                a_t = int(info.get("selected_target", 0))
                choice_target = F.one_hot(torch.tensor(a_t, device=device), num_classes=2).unsqueeze(0).float()
                actor = 1

            # reward index used by the model: 0/1 observed, 2 = no_feedback
            if curr_trial_condition == "AO-o":
                rwd_idx = 2
            else:
                rwd_idx = int(round(float(reward)))
                rwd_idx = 1 if rwd_idx >= 1 else 0

            # key/value tokens (match update2)
            aL = choice_target[..., 0:1]
            aR = choice_target[..., 1:2]
            chosen_feat = aL * left_feats + aR * right_feats

            action_emb = agent.action(torch.tensor([a_t], device=device, dtype=torch.long))     # (1,H)
            actor_emb  = agent.actor(torch.tensor([actor], device=device, dtype=torch.long))    # (1,H)
            rwd_emb    = agent.rwd_in(torch.tensor([rwd_idx], device=device, dtype=torch.long)) # (1,H)
            v_tok = action_emb + actor_emb + rwd_emb  # (1,H)

            k_tok = agent.q_in(chosen_feat)  # (1,H)

            v_tokens.append(v_tok)
            k_tokens.append(k_tok)
            cond_tokens.append(curr_trial_condition)

            # perf stats
            pair_index_ep = info.get("prev_pair_index_in_session", -1)
            selected_high_reward = info.get("selected_high_reward_this_trial", -1)
            sh = int(selected_high_reward) if selected_high_reward is not None else -1
            flipped = info.get("side_is_flipped", False)

            if curr_trial_condition == "IL":
                pair_index_counter_il[pair_index_ep] += 1
            elif curr_trial_condition == "AO-s":
                pair_index_counter_ao[pair_index_ep] += 1
            elif curr_trial_condition == "OL-s":
                pair_index_counter_ol[pair_index_ep] += 1

            if curr_trial_condition == "IL":
                tot_il += 1
                if sh == 1:
                    hi_cnt_il += 1
            elif curr_trial_condition == "AO-s":
                tot_ao += 1
                if sh == 1:
                    hi_cnt_ao += 1
            elif curr_trial_condition == "OL-s":
                tot_ol += 1
                if sh == 1:
                    hi_cnt_ol += 1

            # legacy per-index counters (kept for your per-trial plot)
            if (curr_trial_condition == "IL") and (sh == 1):
                high_reward_choices_il[pair_index_counter_il[pair_index_ep]] += 1
            elif (curr_trial_condition == "AO-s") and (sh == 1):
                high_reward_choice_ao[pair_index_counter_ao[pair_index_ep]] += 1
            elif (curr_trial_condition == "OL-s") and (sh == 1):
                high_reward_choice_ol[pair_index_counter_ol[pair_index_ep]] += 1        


            pair_probs = env.unwrapped.pair_probs
            log(
                "Ep:", info.get("trial_index"),
                "| curr_idx =", pair_index_ep,
                "| iter_il =", pair_index_counter_il[pair_index_ep],
                "| iter_ao =", pair_index_counter_ao[pair_index_ep],
                "| iter_ol =", pair_index_counter_ol[pair_index_ep],
                "| curr_trial_condition =", curr_trial_condition,
                "| choice target:", int(choice_target.argmax(dim=-1).item()),
                "| Reached Target:", info.get("selected_target"),
                "| reward_idx:", rwd_idx,
                "| flipped =", flipped,
                "| selected_high_reward =", selected_high_reward,
                "| p_left =", round(float(p_left), 2),
                "| p_right =", round(float(p_right), 2),
                "| v_old =", round(float(v_old), 2),
                "| delta = ", np.round(delta_val, 2),
                "| g =", np.round(float(g_val), 2),
                "| pair_probs =", [(round(pl, 2), round(pr, 2)) for pl, pr in pair_probs]
            )

            chosen_bandits_buf.append(choice_target.squeeze(0).detach().cpu().numpy())
            bandit_rewards_buf.append(float(rwd_idx))  # store idx so update2 can mask AO-o with 2
            meta_ep_start_buf.append(float(meta_ep_start))
            actor_buf.append(actor)
            trial_cond_buf.append(curr_trial_condition)

            # teacher correctness label (optional supervision)
            tc = -1.0
            if curr_trial_condition in ("AO-o", "OL-o"):
                if selected_high_reward in (-1, None):
                    tc = -1.0
                else:
                    tc = 1.0 if bool(selected_high_reward) else 0.0
            teacher_correct_buf.append(float(tc))

            # next trial
            choice_target = None
            ep_start_flag = 1.0

    bandit_rewards_arr = np.asarray(bandit_rewards_buf, dtype=np.float32)
    trial_cond_arr = np.asarray(trial_cond_buf)

    cum_rewards_il = float(bandit_rewards_arr[trial_cond_arr == "IL"].sum())
    cum_rewards_ao = float(bandit_rewards_arr[trial_cond_arr == "AO-s"].sum())
    cum_rewards_ol = float(bandit_rewards_arr[trial_cond_arr == "OL-s"].sum())

    pct_hi_il = 100.0 * float(hi_cnt_il) / float(max(tot_il, 1))
    pct_hi_ao = 100.0 * float(hi_cnt_ao) / float(max(tot_ao, 1))
    pct_hi_ol = 100.0 * float(hi_cnt_ol) / float(max(tot_ol, 1))

    high_reward_choices_il = np.array(high_reward_choices_il, dtype=np.int32)
    high_reward_choices_ao = np.array(high_reward_choice_ao, dtype=np.int32)
    high_reward_choices_ol = np.array(high_reward_choice_ol, dtype=np.int32)


    if return_obs:
        obs_bandit = {
            "left":  np.stack(left_obs, axis=0),
            "right": np.stack(right_obs, axis=0),
        }
    else:
        obs_bandit = None


    return (
        xy_pos_buf, goal_vec_buf, chosen_bandits_motor_buf,
        obs_bandit, chosen_bandits_buf, bandit_rewards_buf, meta_ep_start_buf,
        actor_buf, trial_cond_buf,
        teacher_correct_buf,
        cum_rewards_il, cum_rewards_ao, cum_rewards_ol,
        pct_hi_il, pct_hi_ao, pct_hi_ol,
        high_reward_choices_il, high_reward_choices_ao, high_reward_choices_ol,
        bandit_logp_buf, bandit_value_buf
    )


# -----------------------------
# Ray worker
# -----------------------------

@ray.remote(num_cpus=1, num_gpus=0)
class RolloutWorker:
    def __init__(self, session_K: int, session_N: int, seed: int = 0, worker_id: int = 0):
        self.session_K = session_K
        self.session_N = session_N
        self.device = torch.device("cpu")
        self.worker_id = worker_id

        self.env = vbe.TwoChoiceReachingEnv(
            W=384,
            H=400,
            render_mode="rgb_array",
            seed=seed,
            session_K=session_K,
            session_N=session_N,
            trial_ms=3000,
            randomize_sides=True,
            shuffle=True,
        )

    def run_session(self, agent_state_dict, probs_this_session, print_this_session: bool = False, teacher_eps: float = 0.0, return_obs: bool = True):
        """Run one session with given pair probs. teacher_eps is optional (only used if env supports it)."""
        hidden_size = 128
        feature_dim = 128
        input_size = feature_dim + 1

        agent = bl.BanditLearner(
            input_size=input_size,
            feature_dim=feature_dim,
            rnn_hidden_size=hidden_size,
            action_dim=2,
            num_pairs=self.session_K,
            max_trials=self.session_N * self.session_K,
        ).to(self.device)

        agent.load_state_dict(agent_state_dict)
        agent.eval()

        self.env.unwrapped.pair_probs = probs_this_session
        if hasattr(self.env.unwrapped, "teacher_eps"):
            try:
                self.env.unwrapped.teacher_eps = float(teacher_eps)
            except Exception:
                pass

        with torch.no_grad():
            return meta_ep_rollout(
                self.env, agent, self.device,
                self.session_K, self.session_N,
                worker_id=self.worker_id,
                print_this_session=print_this_session,
                return_obs=return_obs
            )


# -----------------------------
# Main
# -----------------------------

# bandit_train_batch.py

def _call_update2(agent, optim_bandit, optim_motor,
                 batch_xy_pos, batch_goal_vec, batch_chosen_bandits_motor,
                 batch_bandit_obs, batch_chosen_bandits, batch_bandit_rewards, batch_meta_ep_start,
                 batch_actor, batch_trial_cond, batch_teacher_correct,
                 batch_logp_old, batch_v_old, device):
    return agent.update2(
        optim_bandit, optim_motor,
        batch_xy_pos,
        batch_goal_vec,
        batch_chosen_bandits_motor,
        batch_bandit_obs,
        batch_chosen_bandits,
        batch_bandit_rewards,
        batch_meta_ep_start,
        batch_actor,
        batch_trial_cond,
        batch_teacher_correct,
        batch_logp_old,
        batch_v_old,
        device
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save-dir', type=str, default='checkpoints')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--episodes-per-update', type=int, default=8)  # B
    parser.add_argument('--num-updates', type=int, default=1000)
    parser.add_argument('--warmup-updates', type=int, default=900)
    parser.add_argument('--patience', type=int, default=30)
    parser.add_argument('--min-delta', type=float, default=0.1)
    parser.add_argument('--eval-sessions', type=int, default=200)
    parser.add_argument('--gap-sweep', action='store_true', help='Run gap sweep after training.', default=True)
    parser.add_argument('--p-hi', type=float, default=0.8)
    parser.add_argument('--p-lo-min', type=float, default=0.10)
    parser.add_argument('--p-lo-max', type=float, default=0.6)
    parser.add_argument('--p-lo-steps', type=int, default=14)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # TensorBoard writer
    log_dir = os.path.join("runs", f"bandit_{datetime.now().strftime('%Y%m%d-%H%M%S')}")
    writer = SummaryWriter(log_dir)

    session_K = 3
    session_N = 12

    hidden_size = 128
    feature_dim = 128
    input_size = feature_dim + 1

    # early stopping
    ema = None
    ema_beta = 0.9
    best_ema = -float("inf")
    bad = 0

    agent = bl.BanditLearner(
        input_size=input_size,
        feature_dim=feature_dim,
        rnn_hidden_size=hidden_size,
        action_dim=2,
        num_pairs=session_K,
        max_trials=session_N * session_K,
    )

    optim_bandit = torch.optim.Adam(agent.bandit_parameters(), lr=1e-3)
    optim_motor  = torch.optim.Adam(agent.motor_parameters(),  lr=1e-3)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent.to(device)

    # Ray init
    if not ray.is_initialized():
        tmp_dir = os.path.expanduser("~/tmp/ray")
        os.makedirs(tmp_dir, exist_ok=True)

        ray.init(_temp_dir=tmp_dir, include_dashboard=False, ignore_reinit_error=True)

    # rollout workers
    workers = [
        RolloutWorker.remote(session_K, session_N, seed=args.seed + 1000 * i, worker_id=i)
        for i in range(args.num_workers)
    ]

    num_updates = args.num_updates
    B = args.episodes_per_update
    W = args.num_workers

    # training mixture RNG
    rng_train = np.random.default_rng(args.seed + 123)

    for update_idx in range(num_updates):
        total_sessions_collected = 0

        batch_xy_pos = []
        batch_goal_vec = []
        batch_chosen_bandits_motor = []
        batch_bandit_obs = []
        batch_chosen_bandits = []
        batch_bandit_rewards = []
        batch_meta_ep_start = []
        batch_actor = []
        batch_trial_cond = []
        batch_teacher_correct = []
        batch_actions = []
        batch_logp_old = []
        batch_v_old = []
        batch_policy_mask = []


        cum_rewards_list_il = []
        cum_rewards_list_ao = []
        cum_rewards_list_ol = []

        cum_high_reward_choices_il = []
        cum_high_reward_choices_ao = [] 
        cum_high_reward_choices_ol = []

        p_hi = args.p_hi
        p_lo_min = args.p_lo_min
        p_lo_max = args.p_lo_max

        while total_sessions_collected < B:
            remaining = B - total_sessions_collected
            num_launch = min(W, remaining)

            probs_list = []
            for _ in range(num_launch):
                # Train on a mixture of gaps: p_lo ~ U[p_lo_min, p_lo_max]
                # p_lo = float(rng_train.uniform(p_lo_min, p_lo_max))
                # probs_this_session = sample_probs_this_session(session_K, p_hi, p_lo, rng_train)
                probs_this_session = [(0.8,0.2) if rng_train.random() < 0.5 else (0.2,0.8) for _ in range(session_K)]
                probs_list.append(probs_this_session)

            agent_state_cpu = {k: v.detach().cpu() for k, v in agent.state_dict().items()}

            rollout_futures = []
            for w_id in range(num_launch):
                print_flag = (w_id == 0) and (total_sessions_collected == 0) and (update_idx % 10 == 0)
                rollout_futures.append(
                    workers[w_id].run_session.remote(
                        agent_state_cpu,
                        probs_list[w_id],
                        print_this_session=print_flag
                    )
                )

            rollout_results = ray.get(rollout_futures)

            for res in rollout_results:
                (
                    xy_pos_buf, goal_vec_buf, chosen_bandits_motor_buf,
                    obs_bandit, chosen_bandits_buf, bandit_rewards_buf, meta_ep_start_buf,
                    actor_buf, trial_cond_buf,
                    teacher_correct_buf,
                    cum_rewards_il, cum_rewards_ao, cum_rewards_ol,
                    pct_hi_il, pct_hi_ao, pct_hi_ol,
                    high_reward_choices_il, high_reward_choices_ao, high_reward_choices_ol, logp_old_buf, v_old_buf
                ) = res

                batch_xy_pos.extend(xy_pos_buf)
                batch_goal_vec.extend(goal_vec_buf)
                batch_chosen_bandits_motor.extend(chosen_bandits_motor_buf)
                batch_bandit_obs.append(obs_bandit)
                batch_chosen_bandits.append(torch.as_tensor(np.stack(chosen_bandits_buf), dtype=torch.float32))
                batch_bandit_rewards.append(torch.as_tensor(np.stack(bandit_rewards_buf), dtype=torch.float32))
                batch_meta_ep_start.append(torch.as_tensor(np.stack(meta_ep_start_buf), dtype=torch.float32))
                batch_actor.append(torch.as_tensor(np.stack(actor_buf), dtype=torch.long))
                batch_trial_cond.append(np.stack(trial_cond_buf))
                batch_teacher_correct.append(torch.as_tensor(np.stack(teacher_correct_buf), dtype=torch.float32))
                batch_logp_old.append(torch.as_tensor(np.stack(logp_old_buf), dtype=torch.float32))
                batch_v_old.append(torch.as_tensor(np.stack(v_old_buf), dtype=torch.float32))


                cum_rewards_list_il.append(cum_rewards_il)
                cum_rewards_list_ao.append(cum_rewards_ao)
                cum_rewards_list_ol.append(cum_rewards_ol)

                cum_high_reward_choices_il.append(high_reward_choices_il)
                cum_high_reward_choices_ao.append(high_reward_choices_ao)
                cum_high_reward_choices_ol.append(high_reward_choices_ol)


                total_sessions_collected += 1
                if total_sessions_collected >= B:
                    break

        var_loss, motor_loss = _call_update2(
            agent, optim_bandit, optim_motor,
            batch_xy_pos,
            batch_goal_vec,
            batch_chosen_bandits_motor,
            batch_bandit_obs,
            batch_chosen_bandits,
            batch_bandit_rewards,
            batch_meta_ep_start,
            batch_actor,
            batch_trial_cond,
            batch_teacher_correct,
            batch_logp_old, 
            batch_v_old,
            device
        )

        mean_cum_rew_il = float(np.mean(np.sum(cum_rewards_list_il)))
        mean_cum_rew_ao = float(np.mean(np.sum(cum_rewards_list_ao)))
        mean_cum_rew_ol = float(np.mean(np.sum(cum_rewards_list_ol)))

        print("cum_high_reward_choices_il.shape = ", np.array(cum_high_reward_choices_il).shape)
        print(cum_high_reward_choices_il)


        mean_hi_choices_il = np.mean([arr.sum() for arr in cum_high_reward_choices_il])
        mean_hi_choices_ao = np.mean([arr.sum() for arr in cum_high_reward_choices_ao])
        mean_hi_choices_ol = np.mean([arr.sum() for arr in cum_high_reward_choices_ol])

        #train_score = mean_cum_rew_il + mean_cum_rew_ao + mean_cum_rew_ol
        train_score = mean_hi_choices_il + mean_hi_choices_ao + mean_hi_choices_ol
        ema = train_score if ema is None else (ema_beta * ema + (1 - ema_beta) * train_score)

        if update_idx % 10 == 0:
            print(f"[upd {update_idx:04d}] var_loss={var_loss:.4f} motor_loss={motor_loss:.4f} "
                  f"mean_hi_choices_il={mean_hi_choices_il:.1f} mean_hi_choices_ao={mean_hi_choices_ao:.1f} mean_hi_choices_ol={mean_hi_choices_ol:.1f} "
                  f"mean_cum_rew_il={mean_cum_rew_il:.1f} mean_cum_rew_ao={mean_cum_rew_ao:.1f} mean_cum_rew_ol={mean_cum_rew_ol:.1f}")

        # Early stopping (after warmup)
        if update_idx >= args.warmup_updates:
            if ema > best_ema + args.min_delta:
                best_ema = ema
                bad = 0
                save_checkpoint(
                    os.path.join(args.save_dir, "best.pt"),
                    agent, optim_bandit, optim_motor,
                    extra={"update": update_idx, "ema_score": float(ema)}
                )
            else:
                bad += 1

            if bad >= args.patience:
                print(f"Early stopping at update {update_idx}, best_ema={best_ema:.2f}")
                break

        # TensorBoard
        writer.add_scalar("Loss/ChoiceLoss", float(var_loss), update_idx)
        writer.add_scalar("Loss/MotorLoss", float(motor_loss), update_idx)
        writer.add_scalar("Reward/MeanCumSession_IL", mean_cum_rew_il, update_idx)
        writer.add_scalar("Reward/MeanCumSession_AO", mean_cum_rew_ao, update_idx)
        writer.add_scalar("Reward/MeanCumSession_OL", mean_cum_rew_ol, update_idx)

    # -----------------------------
    # Load best checkpoint
    # -----------------------------
    ckpt_path = os.path.join(args.save_dir, "best.pt")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    print(f"Loaded best checkpoint from update {ckpt['extra']['update']}, ema_score={ckpt['extra']['ema_score']:.2f}")
    agent.load_state_dict(ckpt["model_state"])
    agent.eval()

    # -----------------------------
    # Fixed-prob eval + per-trial plot
    # -----------------------------
    eval_il = []
    eval_ao = []
    eval_ol = []

    mean_cum_rew_il = 0.0
    mean_cum_rew_ao = 0.0
    mean_cum_rew_ol = 0.0

    rng_eval = np.random.default_rng(args.seed + 999)
    agent_state_cpu = {k: v.detach().cpu() for k, v in agent.state_dict().items()}

    probs_eval = [(0.2, 0.8) if rng_eval.random() < 0.5 else (0.8, 0.2) for _ in range(session_K)]

    futures = []
    for i in range(args.eval_sessions):
        futures.append(
            workers[i % len(workers)].run_session.remote(
                agent_state_cpu,
                probs_this_session=probs_eval,
                print_this_session=False,
                return_obs=False
            )
        )
    results = ray.get(futures)

    for res in results:
        (
            _, _, _,
            _, _, _, _,
            _, _,
            _,  # teacher_correct_buf
            cum_rew_il, cum_rew_ao, cum_rew_ol,
            pct_hi_il, pct_hi_ao, pct_hi_ol,
            high_reward_choices_il,
            high_reward_choices_ao,
            high_reward_choices_ol,
            _, _
        ) = res

        eval_il.append(high_reward_choices_il)
        eval_ao.append(high_reward_choices_ao)
        eval_ol.append(high_reward_choices_ol)

        mean_cum_rew_il += cum_rew_il
        mean_cum_rew_ao += cum_rew_ao
        mean_cum_rew_ol += cum_rew_ol

    mean_il = np.array(eval_il).mean(axis=0)
    mean_ao = np.array(eval_ao).mean(axis=0)
    mean_ol = np.array(eval_ol).mean(axis=0)

    mean_cum_rew_il /= args.eval_sessions
    mean_cum_rew_ao /= args.eval_sessions
    mean_cum_rew_ol /= args.eval_sessions

    print(f"Eval results over {args.eval_sessions} sessions:")
    print(f" Mean cum reward IL: {mean_cum_rew_il:.1f}, AO: {mean_cum_rew_ao:.1f}, OL: {mean_cum_rew_ol:.1f}")
    print(f" Mean high-reward choices per session IL: {mean_il.sum():.1f}, AO: {mean_ao.sum():.1f}, OL: {mean_ol.sum():.1f}")

    plot_dir = os.path.join(args.save_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    run_dir = os.path.join(plot_dir, datetime.now().astimezone().strftime("run_%Y%m%d_%H%M%S"))
    os.makedirs(run_dir, exist_ok=True)


    fig, ax = plt.subplots(figsize=(6, 4))
    trials = np.arange(session_N)
    ax.plot(trials, mean_il, label="IL")
    ax.plot(trials, mean_ao, label="AO")
    ax.plot(trials, mean_ol, label="OL")
    ax.set_xlabel("Trial index")
    ax.set_ylabel("Mean high-reward choice")
    ax.set_title("Per-trial performance (best checkpoint)")
    ax.legend()
    ax.grid(True)

    plot_path = os.path.join(run_dir, "PerTrialReward.png")
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    writer.add_figure("Eval/PerTrialReward", fig)
    plt.close(fig)

    # -----------------------------
    # Gap sweep (optional)
    # -----------------------------
    if args.gap_sweep:
        rng = np.random.default_rng(args.seed + 2026)

        p_hi = float(args.p_hi)
        p_lo_grid = np.linspace(float(args.p_lo_max), float(args.p_lo_min), int(args.p_lo_steps))
        sweep = []

        for p_lo in p_lo_grid:
            # mean high-arm pick rate (%), averaged over eval sessions
            pct_il = 0.0
            pct_ao = 0.0
            pct_ol = 0.0

            futures = []
            print(f"[GapSweep] launching p_lo={p_lo:.2f} ...")
            for i in range(args.eval_sessions):
                probs_this_session = sample_probs_this_session(session_K, p_hi, float(p_lo), rng)
                futures.append(
                    workers[i % len(workers)].run_session.remote(
                        agent_state_cpu,
                        probs_this_session=probs_this_session,
                        print_this_session=False,
                        return_obs=False
                    )
                )

            results = ray.get(futures)

            for res in results:
                (
                    _, _, _,
                    _, _, _, _,
                    _, _,
                    _,  # teacher_correct_buf
                    _, _, _,              # cum_rew_il, cum_rew_ao, cum_rew_ol (unused)
                    pct_hi_il, pct_hi_ao, pct_hi_ol,
                    _, _,                 # high_reward_choices_* (unused here)
                    _, _, _
                ) = res

                pct_il += float(pct_hi_il)
                pct_ao += float(pct_hi_ao)
                pct_ol += float(pct_hi_ol)

            pct_il /= args.eval_sessions
            pct_ao /= args.eval_sessions
            pct_ol /= args.eval_sessions

            sweep.append({
                "p_lo": float(p_lo),
                "gap": float(p_hi - p_lo),
                "pct_hi_il": float(pct_il),
                "pct_hi_ao": float(pct_ao),
                "pct_hi_ol": float(pct_ol),
            })

            print(f"[GapSweep] p_lo={p_lo:.2f} gap={p_hi-p_lo:.2f} | IL={pct_il:.1f}% AO={pct_ao:.1f}% OL={pct_ol:.1f}%")

        # Save CSV
        # Save CSV
        csv_path = os.path.join(run_dir, "GapSweep.csv")
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write("p_lo,gap,pct_hi_il,pct_hi_ao,pct_hi_ol\n")
            for row in sweep:
                f.write(f"{row['p_lo']:.4f},{row['gap']:.4f},{row['pct_hi_il']:.4f},{row['pct_hi_ao']:.4f},{row['pct_hi_ol']:.4f}\n")

        gaps   = np.array([d["gap"] for d in sweep])
        pct_il = np.array([d["pct_hi_il"] for d in sweep])
        pct_ao = np.array([d["pct_hi_ao"] for d in sweep])
        pct_ol = np.array([d["pct_hi_ol"] for d in sweep])

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(gaps, pct_il, label="IL")
        ax.plot(gaps, pct_ao, label="AO")
        ax.plot(gaps, pct_ol, label="OL")
        ax.set_xlabel("Gap (p_hi - p_lo)")
        ax.set_ylabel("High-arm pick rate (%)")
        ax.set_title("Gap sweep (high-arm accuracy)")
        ax.grid(True)
        ax.legend()
        fig.savefig(os.path.join(run_dir, "GapSweep_HighArmPct.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)

        # optional: separation plot
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(gaps, pct_ol - pct_ao, label="OL - AO")
        ax.plot(gaps, pct_ao - pct_il, label="AO - IL")
        ax.axhline(0, linewidth=1)
        ax.set_xlabel("Gap (p_hi - p_lo)")
        ax.set_ylabel("Δ high-arm pick rate (%)")
        ax.set_title("Separation vs difficulty")
        ax.grid(True)
        ax.legend()
        fig.savefig(os.path.join(run_dir, "GapSweep_DeltasPct.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)

    # Cleanup
    ray.shutdown()
    writer.close()


if __name__ == "__main__":
    main()
