# DiffChir : Automatisation du calcul de diffraction des surfaces chirales -- PAS FINI

**DiffChir** est un framework modulaire en Python conçu pour automatiser la simulation électromagnétique, l'analyse polarimétrique complète et l'optimisation inverse des métasurfaces et réseaux de diffraction chiraux.

---

## Sommaire
- [Présentation et Contexte Physique](#présentation-et-contexte-physique)
- [Fonctionnalités Clés](#fonctionnalités-clés)
- [Architecture du Pipeline](#architecture-du-pipeline)
- [Installation](#installation)
- [Utilisation Rapide](#utilisation-rapide)
- [Formulation Théorique Implémentée](#formulation-théorique-implémentée)
- [Passage à l'Échelle et Optimisation Inverse](#passage-à-léchelle-et-optimisation-inverse)
- [Structure du Dépôt](#structure-du-dépôt)
- [Licence](#licence)

---

## Présentation et Contexte Physique

Les surfaces chirales (gammadions, structures en L/Z, hélices 3D ou bicouches tournées) brisent les symétries miroir et d'inversion spatiale. En conséquence, elles interagissent différemment avec la lumière polarisée circulairement à gauche (**LCP**) et à droite (**RCP**), générant des effets de **dichroïsme circulaire ($CD$)** et d'**activité optique** distribués dans les différents ordres de diffraction $(m, n)$.

Parce que la brisure de symétrie interdit toute réduction du domaine de calcul par des conditions aux limites symétriques ou antisymétriques, la simulation de la maille élémentaire complète en 3D est requise. **DiffChir** automatise l'ensemble de ce workflow : de la génération paramétrique de la maille jusqu'au stockage structuré des matrices de Jones et de Mueller.

---

## Fonctionnalités Clés

- **Indépendance vis-à-vis du solveur :** Interface unifiée compatible avec les solveurs RCWA/FMM (`torcwa`, `fmmax`, `S4`), FDTD (`Meep`, `Tidy3D`) et FEM (`COMSOL`).
- **Analyse polarimétrique multi-ordres :** Extraction automatique des matrices de Jones $2 \times 2$ en bases linéaire $(s,p)$ et circulaire $(L,R)$, et conversion vers les matrices de Mueller $4 \times 4$.
- **Observables chirales :** Calcul direct des efficacités de diffraction ($T_{LL}, T_{LR}, T_{RL}, T_{RR}$), du dichroïsme circulaire en transmission ($CD_T$) et en réflexion ($CD_R$), ainsi que de la rotation azimutale ($\Psi$) et de l'ellipticité ($\chi$).
- **Validation numérique automatique :** Contrôle intégré de la convergence (nombre d'harmoniques de Fourier ou pas de maillage) et vérification stricte de la conservation de l'énergie ($\sum (R + T) + A = 1$).
- **Prêt pour le HPC et l'optimisation inverse :** Vectorisation GPU par lots (*batching*) et support des gradients par méthode adjointe / différentiation automatique.

---

## Architecture du Pipeline

```text
+-------------------------------------------------------------------+
| 1. Générateur Géométrique & Matériaux (Paramètres p, n(λ))         |
+---------------------------------+---------------------------------+
                                  |
                                  v
+-------------------------------------------------------------------+
| 2. Contrôleur de Convergence (Balayage résolution / harmoniques)   |
+---------------------------------+---------------------------------+
                                  |
                                  v
+-------------------------------------------------------------------+
| 3. Solveur Électromagnétique (Excitations orthogonales s et p)    |
+---------------------------------+---------------------------------+
                                  |
                                  v
+-------------------------------------------------------------------+
| 4. Post-Traitement Polarimétrique (Jones, Mueller, CD, Poynting)  |
+---------------------------------+---------------------------------+
                                  |
                                  v
+-------------------------------------------------------------------+
| 5. Stockage Structuré (HDF5) & Boucle d'Optimisation Inverse      |
+-------------------------------------------------------------------+
```

---

## Installation

```bash
git clone https://github.com/votre-utilisateur/DiffChir.git
cd DiffChir
pip install -r requirements.txt
```

### Dépendances recommandées
- **Coeur :** `numpy`, `scipy`, `h5py`, `pydantic`
- **Solveurs RCWA différentiables (optionnel) :** `torch`, `torcwa` ou `jax`, `fmmax`
- **Parallélisation :** `joblib`, `mpi4py`

---

## Utilisation Rapide

L'exemple ci-dessous montre comment configurer une surface périodique chirale, analyser la sortie d'un solveur et extraire le dichroïsme circulaire par ordre de diffraction :

```python
import numpy as np
from chiral_diffraction import ChiralSurfaceConfig, DiffChirAnalyzer

# 1. Configuration de la maille et de l'illumination
config = ChiralSurfaceConfig(
    period_x=500.0,        # Période selon x (nm)
    period_y=500.0,        # Période selon y (nm)
    wavelength=650.0,      # Longueur d'onde incidente (nm)
    n_superstrate=1.0,     # Milieu incident (Air)
    n_substrate=1.45,      # Substrat (Silice)
    theta_inc_deg=0.0,     # Incidence normale
    phi_inc_deg=0.0,
    fourier_orders=(9, 9)
)

# 2. Initialisation de l'analyseur polarimétrique
analyzer = DiffChirAnalyzer(config)

# 3. Exemple de matrices de Jones linéaires J_T et J_R issues d'un solveur pour l'ordre (0, 0)
J_T_lin = np.array([[0.65 + 0.1j, 0.12 - 0.08j],
                    [-0.12 + 0.08j, 0.65 + 0.1j]], dtype=np.complex128)
J_R_lin = np.array([[0.15 + 0.02j, 0.01 + 0.01j],
                    [-0.01 - 0.01j, 0.15 + 0.02j]], dtype=np.complex128)

solver_output = {(0, 0): (J_T_lin, J_R_lin)}

# 4. Extraction des observables chirales et vérification du bilan d'énergie
results = analyzer.run_full_analysis(solver_output)

order_00 = results["orders"][(0, 0)]
print(f"Transmission LCP (T_L) : {order_00['T_L']:.4f}")
print(f"Transmission RCP (T_R) : {order_00['T_R']:.4f}")
print(f"Dichroïsme Circulaire (CD_T) : {order_00['CD_T']:.4f}")
print(f"Bilan énergétique (LCP) : {results['energy_balance_L']:.4f}")
```

---

## Formulation Théorique Implémentée

### 1. Changement de base Linéaire $\leftrightarrow$ Circulaire
En convention harmonique $\exp(-i\omega t)$, la matrice de Jones dans la base circulaire $(L, R)$ est obtenue par similitude unitaire à partir de la base linéaire locale $(s, p)$ de chaque ordre $(m,n)$ :

$$\mathbf{J}_{\text{circ}}^{(m,n)} = \mathbf{\Lambda}^{-1} \mathbf{J}_{\text{lin}}^{(m,n)} \mathbf{\Lambda}, \qquad \mathbf{\Lambda} = \frac{1}{\sqrt{2}} \begin{pmatrix} 1 & 1 \\ i & -i \end{pmatrix}$$

### 2. Efficacités de diffraction et Dichroïsme Circulaire
Les efficacités de diffraction sont pondérées par le rapport des composantes longitudinales des vecteurs d'onde propagatifs :

$$T_{uv}^{(m,n)} = \frac{\text{Re}\left(k_{z,2}^{(m,n)}\right)}{k_{z,1}^{(0,0)}} \left| t_{uv}^{(m,n)} \right|^2, \qquad u, v \in \{L, R\}$$

Le dichroïsme circulaire normalisé en transmission pour l'ordre $(m,n)$ est défini par :

$$CD_T^{(m,n)} = \frac{\left(T_{LL}^{(m,n)} + T_{RL}^{(m,n)}\right) - \left(T_{RR}^{(m,n)} + T_{LR}^{(m,n)}\right)}{\left(T_{LL}^{(m,n)} + T_{RL}^{(m,n)}\right) + \left(T_{RR}^{(m,n)} + T_{LR}^{(m,n)}\right)}$$

---

## Passage à l'Échelle et Optimisation Inverse

- **Parallélisation CPU / GPU :** Exécution multi-nœuds via `joblib` / `mpi4py` (avec `OMP_NUM_THREADS=1` pour éviter la contention BLAS) ou vectorisation par lots de longueurs d'onde sur GPU via PyTorch/JAX.
- **Optimisation Topologique Adjointe :** Grâce au théorème de réciprocité de Lorentz, le gradient d'une fonction objectif $\mathcal{F}$ (ex. maximisation de $CD_T^{(+1,0)}$) par rapport à la carte de permittivité $\varepsilon(\mathbf{r})$ est calculé en exactement **deux simulations** (directe et adjointe) par itération, couplées à des filtres morphologiques garantissant la fabricabilité.

---

## Structure du Dépôt

```text
DiffChir/
├── chiral_diffraction/
│   ├── __init__.py
│   ├── config.py          # Validation géométrique et dispersion des matériaux
│   ├── analyzer.py        # Calcul des matrices de Jones, Mueller, CD et Poynting
│   ├── convergence.py     # Algorithmes automatiques de test de convergence
│   └── solvers/           # Adaptateurs pour RCWA (torcwa/fmmax), FDTD et FEM
├── examples/
│   ├── twisted_bilayer_rcwa.py
│   └── inverse_design_chiral_grating.py
├── docs/
│   └── diffraction_surfaces_chirales.tex  # Documentation théorique complète en LaTeX
├── tests/
│   └── test_energy_conservation.py
├── requirements.txt
└── README.md
```

---

## Licence

on verra plus tard
