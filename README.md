# Liquidity Providing Automated Market Making

Simulation RL Uniswap V3 (simplifiee) avec Gymnasium + Ray RLlib.

## Notes de modelisation

- Le moteur reproduit les mecaniques essentielles: ticks, liquidite concentree, fees actives.
- Le modele est volontairement simplifie pour prototypage RL:
  - une seule position LP par agent,
  - dynamique de prix type GBM,
  - volume externe simule.
- Reward: variation de valeur de portefeuille d'un step a l'autre.
