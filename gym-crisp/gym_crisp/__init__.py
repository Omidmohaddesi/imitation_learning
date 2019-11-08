import logging
from gym.envs.registration import register

logger = logging.getLogger(__name__)

register(
    id='Crisp-v0',
    entry_point='gym_crisp.envs:CrispEnv',
)
