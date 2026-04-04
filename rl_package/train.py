"""
Train an RL policy for cart person-following.

Usage:
    # Train in simulation (no hardware needed):
    python -m cart_stack.rl.train --backend sim --timesteps 200000

    # Train on real Pi:
    python -m cart_stack.rl.train --backend real --pi-url http://172.20.10.4:8001 --timesteps 50000

    # Resume training from checkpoint:
    python -m cart_stack.rl.train --resume cart_follow_policy.zip --timesteps 100000
"""

from __future__ import annotations

import argparse
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from cart_stack.rl.env import CartFollowEnv


def make_env(backend: str, pi_url: str, target_gap: float):
    def _init():
        return CartFollowEnv(backend=backend, pi_url=pi_url, target_gap_m=target_gap)
    return _init


def train(args):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Create vectorized environment
    env = DummyVecEnv([make_env(args.backend, args.pi_url, args.target_gap)])
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    # Eval env (always sim for consistency)
    eval_env = DummyVecEnv([make_env("sim", "", args.target_gap)])
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    if args.resume:
        print(f"Resuming from {args.resume}")
        model = PPO.load(args.resume, env=env)
    else:
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            verbose=1,
            tensorboard_log=str(out_dir / "tb_logs"),
        )

    # Callbacks
    checkpoint_cb = CheckpointCallback(
        save_freq=10000,
        save_path=str(out_dir / "checkpoints"),
        name_prefix="cart_follow",
    )
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=str(out_dir / "best_model"),
        log_path=str(out_dir / "eval_logs"),
        eval_freq=5000,
        n_eval_episodes=10,
        deterministic=True,
    )

    print(f"Training for {args.timesteps} timesteps on {args.backend} backend...")
    model.learn(
        total_timesteps=args.timesteps,
        callback=[checkpoint_cb, eval_cb],
    )

    # Save final model
    model_path = out_dir / "cart_follow_policy"
    model.save(str(model_path))
    env.save(str(out_dir / "vec_normalize.pkl"))
    print(f"Saved model to {model_path}.zip")
    print(f"Saved normalization stats to {out_dir / 'vec_normalize.pkl'}")


def main():
    parser = argparse.ArgumentParser(description="Train cart-following RL policy")
    parser.add_argument("--backend", choices=["sim", "real"], default="sim")
    parser.add_argument("--pi-url", default="http://172.20.10.4:8001")
    parser.add_argument("--target-gap", type=float, default=0.65)
    parser.add_argument("--timesteps", type=int, default=200_000)
    parser.add_argument("--resume", type=str, default=None, help="Path to .zip model to resume")
    parser.add_argument("--out-dir", default="rl_output")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
