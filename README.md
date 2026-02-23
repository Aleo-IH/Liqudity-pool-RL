# Pool de liquidtés


## L'idée de base : Qu'est-ce qu'une Liquidity Pool ?

Imaginez un bureau de change à l'aéroport. Si vous voulez échanger des Euros contre des Dollars, le bureau doit avoir un coffre-fort contenant à la fois des Euros et des Dollars pour faire l'échange.

Dans la cryptomonnaie, il n'y a pas d'entreprise centralisée pour gérer ce coffre-fort. À la place, on utilise un programme informatique autonome (un *Smart Contract*) appelé **Liquidity Pool**.

### Les parties prenantes (Stakeholders)

* **Les Fournisseurs de Liquidité (LPs) :** Ce sont des utilisateurs normaux qui décident de prêter leurs propres cryptomonnaies (par exemple, du Bitcoin et de l'Ethereum) au "coffre-fort". En échange, ils gagnent de l'argent.
* **Les Traders :** Ce sont les personnes qui utilisent la pool pour échanger une crypto contre une autre. Ils paient une petite commission (frais de transaction) à chaque échange.
* **Le Protocole (Uniswap) :** C'est l'infrastructure technique qui met en relation les LPs et les Traders de manière automatique et sécurisée.

**Le deal est simple :** Les LPs fournissent l'argent, les Traders l'utilisent pour faire leurs échanges et paient une commission, et cette commission est reversée aux LPs comme récompense.


## La révolution d'Uniswap V3 : La "Liquidité Concentrée"

Dans les anciennes versions (Uniswap V2), quand un LP mettait son argent dans le coffre, cet argent était étalé sur **tous les prix possibles** (de 0$ à l'infini). C'était très inefficace car la plupart de l'argent "dormait" et n'était jamais utilisé pour les échanges réels.

**Uniswap V3 a introduit la Liquidité Concentrée.** En tant que LP, vous pouvez désormais dire : *"Je veux que mon argent soit utilisé uniquement quand le prix de l'Ethereum se situe entre 3000$ et 4000$."* * **Avantage :** Votre argent est utilisé beaucoup plus souvent (car il est placé là où se passent les vrais échanges), ce qui génère beaucoup plus de frais pour vous (jusqu'à 4000 fois plus efficace).

* **Inconvénient :** Si le prix sort de votre fourchette (ex: l'Ethereum tombe à 2900$), votre argent n'est plus utilisé, vous ne touchez plus de frais, et vous devez réajuster votre position manuellement.



## Les mathématiques derrière la machine (Simplifiées)

Pour que le programme informatique décide tout seul du prix d'une monnaie sans qu'aucun humain n'intervienne, il utilise des formules mathématiques (ce qu'on appelle un *Automated Market Maker* ou AMM).

### La formule d'équilibre (Le produit constant)

La règle d'or de base de ces pools est dictée par une équation très simple :
$$
x \cdot y = k
$$

* $x$ = la quantité de la première crypto (ex: Bitcoin)
* $y$ = la quantité de la deuxième crypto (ex: Ethereum)
* $k$ = une constante qui ne change pas pendant l'échange.

**Comment ça marche ?** Si un trader achète beaucoup de Bitcoin ( diminue dans le coffre), le programme va automatiquement augmenter le prix du Bitcoin pour que  reste constant. C'est la loi de l'offre et de la demande, mais gérée par une équation.

On a donc 
$$
P = \frac{x}{y}
$$
le prix de la crypto en question (sur une paire **ETHUSDC** le prix de **ETH** en **USDC**)

### Calculer son dépôt (Les montants par fourchette de prix)

Puisque dans la V3 vous concentrez votre argent entre un prix minimum $P_a$ et un prix maximum $P_b$, Uniswap utilise des racines carrées pour calculer la quantité exacte de chaque token (représentée par  et ) que vous devez déposer pour une liquidité  donnée :

$$
\Delta x = 
$$

*Concrètement : Le code calcule à votre place la proportion exacte de Bitcoin et d'Ethereum dont vous avez besoin selon la taille de la fourchette de prix que vous avez choisie.*

### Le système de "Ticks" (Les crans de prix)

Pour que l'ordinateur puisse calculer tout cela rapidement, la ligne des prix n'est pas continue, elle est divisée en petits crans appelés **Ticks** (comme les graduations d'une règle). Le prix  à un Tick précis est calculé par :

Chaque fois que le prix d'une monnaie change de 0.01%, on passe au "Tick" suivant.

---

## Le grand risque : La Perte Éphémère (Impermanent Loss)

C'est le concept le plus contre-intuitif mais le plus important. Quand vous placez vos cryptos dans une pool, leur quantité va fluctuer en fonction des achats et ventes des traders.

Si le prix d'une de vos cryptos explose à la hausse ou s'effondre à la baisse par rapport à son prix d'origine (), la valeur totale de votre dépôt sera **inférieure** à ce qu'elle aurait été si vous aviez simplement gardé vos cryptos bien sagement dans votre portefeuille sans rien faire.

L'équation qui mesure cette perte en pourcentage est :

*  = le prix lors de votre dépôt.
*  = le prix actuel sur le marché.

## Exemple
Pour simplifier, nous allons utiliser une pool classique (où la liquidité est répartie partout) avec deux monnaies :

1. L'**Ethereum (ETH)**, dont le prix varie.
2. L'**USDC**, une crypto stable qui vaut toujours **1€**.

Voici comment vos **100€** vont évoluer étape par étape.

---

### Le point de départ (Votre dépôt)

Pour entrer dans une pool classique, vous devez fournir une valeur égale des deux monnaies (50/50).
Imaginons qu'aujourd'hui, **1 ETH vaut 2000€**.

Pour investir vos 100€, vous les coupez en deux :

* Vous déposez **50 USDC** (qui valent 50€).
* Vous achetez et déposez **0,025 ETH** (qui valent 50€ au prix de 2000€).

Votre valeur totale de départ est bien de **100€**.

---

### Le marché est calme (Vous encaissez les frais)

Pendant un mois, le prix de l'Ethereum reste stable autour de 2000€. Pendant ce temps, des dizaines de traders utilisent la pool pour échanger des ETH contre des USDC, et inversement.

* À chaque transaction, ils paient une petite taxe (par exemple 0,3%).
* Comme vous prêtez votre argent au "coffre-fort", vous touchez une partie de cette taxe.
* Au bout d'un mois, disons que vous avez récolté l'équivalent de **2€ de frais**.

**Bilan de l'étape 2 :** Vous avez toujours vos 50 USDC et vos 0,025 ETH, mais vous avez gagné **2€** en plus. Votre investissement est rentable.

---

### Le prix de l'Ethereum explose (La Perte Éphémère)

Soudain, une excellente nouvelle tombe sur le marché de la crypto et le prix de l'Ethereum double : **1 ETH vaut maintenant 4000€**.

C'est ici que la magie (et le piège) du contrat intelligent opère. Rappelez-vous, la pool cherche toujours à s'équilibrer. Comme l'ETH devient très cher et très demandé, les traders vont en acheter dans la pool en y injectant des USDC. Le contrat intelligent va donc **vendre automatiquement une partie de vos ETH en échange d'USDC**.

Votre panier de cryptos dans la pool a changé. Grâce aux mathématiques de la pool, vous possédez maintenant :

* **70,71 USDC** (vous en avez plus qu'au début)
* **0,0176 ETH** (vous en avez moins qu'au début)

Calculons la valeur de votre nouveau panier avec le nouveau prix de l'ETH (4000€) :

* Vos USDC valent toujours **70,71€**.
* Vos 0,0176 ETH valent désormais **70,68€** (0,0176 multiplié par 4000).

Votre valeur totale dans la pool est de **141,39€**. Vous avez fait un beau bénéfice par rapport à vos 100€ de départ !

**Mais alors, où est la "Perte" ?**
La perte éphémère ne se calcule pas par rapport à votre mise de départ, mais par rapport à ce que vous auriez eu **si vous n'aviez rien fait** (si vous aviez gardé vos cryptos dans votre portefeuille sans les mettre dans la pool).

| Scénario | Calcul avec l'ETH à 4000€ | Valeur Totale |
| --- | --- | --- |
| **Garder dans son portefeuille (Rien faire)** | 50 USDC + (0,025 ETH x 4000) | **150,00€** |
| **Investir dans la Liquidity Pool** | 70,71 USDC + (0,0176 ETH x 4000) | **141,39€** |

La différence entre les deux est de **8,61€**. C'est cela, la Perte Éphémère (Impermanent Loss). Le contrat a vendu vos ETH pendant qu'ils montaient, limitant ainsi vos gains potentiels.

---

### Le Bilan Final

Si vous retirez votre argent de la pool à ce moment précis, voici votre résultat réel :

* Valeur de vos cryptos dans la pool : **141,39€**
* Les frais de trading que vous avez gagnés : **+ 2,00€**
* **Total en poche : 143,39€**

Vous êtes largement gagnant par rapport à vos 100€ du début. Cependant, vous êtes un peu déçu, car si vous aviez simplement gardé vos cryptos sans jouer au fournisseur de liquidité, vous auriez eu 150€.

Fournir de la liquidité est donc un pari : vous pariez que le marché ne va pas trop bouger violemment dans un sens ou dans l'autre, et que les **frais de trading** que vous allez accumuler compenseront largement cette petite perte d'opportunité.

Voulez-vous que l'on applique cet exemple à la fameuse "Liquidité Concentrée" de la V3 pour voir comment vous auriez pu générer beaucoup plus que 2€ de frais dans ce même scénario ?