from __future__ import annotations

import numpy as np


def generate_channel_trajectories(
    n_trajectories: int = 300,
    n_steps: int = 900,
    dt: float = 1.0,
    diffusivity: float = 4e-5,
    seed: int | None = 11,
) -> list[np.ndarray]:
    """Generate toy advective-diffusive trajectories with speed memory.

    This is not a porous-media simulator. It is a controlled stand-in with
    persistent fast channels, slow zones, transverse confinement, and diffusion.
    That gives us something non-Fickian enough to test the TTA mechanics before
    real DNS trajectories are dropped in.
    """
    rng = np.random.default_rng(seed)
    trajectories: list[np.ndarray] = []

    for _ in range(n_trajectories):
        path = np.zeros((n_steps + 1, 2), dtype=float)
        path[0, 1] = rng.uniform(-0.8, 0.8)

        mobile = rng.random() < 0.7
        log_speed = np.log(0.018 if mobile else 0.004)
        angle = rng.normal(0.0, 0.15)

        for step in range(n_steps):
            if rng.random() < (0.006 if mobile else 0.018):
                mobile = not mobile

            target_speed = 0.031 if mobile else 0.0045
            log_speed = 0.985 * log_speed + 0.015 * np.log(target_speed)
            log_speed += rng.normal(0.0, 0.035 if mobile else 0.055)
            speed = float(np.exp(log_speed))

            y = path[step, 1]
            channel_bias = 0.18 * np.sin(1.7 * path[step, 0] + 2.2 * y)
            angle = 0.94 * angle + 0.06 * channel_bias + rng.normal(0.0, 0.035)

            advective = np.array(
                [
                    speed * np.cos(angle),
                    0.35 * speed * np.sin(angle) - 0.0025 * y,
                ]
            )
            diffusive = np.sqrt(2.0 * diffusivity * dt) * rng.normal(size=2)
            path[step + 1] = path[step] + dt * advective + diffusive

            if path[step + 1, 1] > 1.0:
                path[step + 1, 1] = 2.0 - path[step + 1, 1]
            elif path[step + 1, 1] < -1.0:
                path[step + 1, 1] = -2.0 - path[step + 1, 1]

        trajectories.append(path)

    return trajectories

