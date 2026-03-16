# ---
# title: Titre du travail
# repository: tpoisot/BIO245-modele
# auteurs:
#    - nom: Nadler
#      prenom: Christina
#      matricule: 20313890
#      github: ChristinaN31
#    - nom: Gomez Saucedo
#      prenom: Carla Danahe
#      matricule: 20341379
#      github: CarlaGomez1
# ---

# # Introduction
# Les corridors sous les lignes électriques à haute tension présentent à 
# la fois des enjeux écologiques et des contraintes de sécurité 
# (clarke2008towardsecologicalmanagement). En effet, la croissance d'arbres et de
# végétation de grande taille peut interférer avec les infrastructures
# électriques, augmentant ainsi les risques de pannes de courant,
# d'incendies, ou encore d’endommagement du matériel. Il est donc
# nécessaire de maintenir ces corridors libres de végétation afin d’assurer le bon 
# fonctionnement de ses installations. Cependant, la gestion de ces 
# corridors doit tenir compte des contraintes écologiques, car ils 
# représentent une niche écologique importante pour certains animaux. 
# C’est donc très important de préserver cette biodiversité tant animale 
# que végétale afin d’éviter leur disparition et d’éviter de dégrader ces
# milieux de façon irréversible (clarke2008towardsecologicalmanagement). On ne peut 
# donc pas prendre en considération l’option de raser l’entièreté de la 
# végétation. Une approche couramment utilisée pour approcher cette
# problématique est la gestion intégrée de la végétation, qui vise à
# favoriser la croissance de plantes de plus petite taille, comme des
# herbes ou encore des buissons (kimmel2024integratedvegetationmanagement)
# tout en limitant l’installation d’arbres susceptibles de devenir assez 
# grands pour toucher les lignes électriques. Dans cette simulation, les 
# arbres ou autre végétation pouvant interférer avec les lignes électriques ont été coupés, 
# il n’existe donc que des parcelles pouvant se trouver dans quatre stades
# possibles : vide, avec des herbes, avec des buissons de type A ou de 
# type B. On cherche donc à étudier l’évolution de cette communauté
# végétale au fil des générations au moyen des transitions des parcelles
# entre ses quatre états, sous une ligne électrique en tenant compte de
# contraintes de sécurité et de conservation. La question de ce travail
# est alors la suivante : est-ce que l’introduction de deux espèces de 
# buissons (type A et type B), permet-elle de créer une communauté 
# végétale stable sous un corridor électrique qui limiterait les accidents
# et maintiendrait un certain niveau de biodiversité? La première 
# hypothèse posée pour ce modèle est que les deux espèces de buisson
# pourraient coexister ou se remplacer dans les parcelles, ce qui 
# contribuerait à maintenir une couverture végétale basse tout en 
# augmentant la diversité végétale dans les parcelles En effet, la 
# présence de deux types (A et B) de buissons, pourrait faciliter la 
# colonisation des parcelles disponibles. Si certaines conditions
# environnementales sont moins favorables pour une des espèces, l’autre
# pourrait tout de même s’y établir. Au contraire, si les conditions sont
# favorables dans les parcelles pour les deux types de buissons, ils 
# pourraient coexister dans le paysage. Dans cette simulation, on
# s’attend donc au fait que l'ajout d'un deuxième type de buisson puisse contribuer à 
# créer une communauté végétale plus stable et diversifiée, tout en respectant les contraintes 
# de sécurité imposées par la présence des lignes électriques. 

# # Présentation du modèle
# ## Description du  modèle :
# Dans ce modèle, le corridor sous une ligne électrique est représenté par un ensemble de parcelles
# indépendantes. Chaque parcelle correspond à une portion du paysage pouvant être occupée
# par différents types de végétation (buissons ou herbacées). L'objectif du modèle est donc d'observer comment la végétation peut
# évoluer dans le temps sous l'effet de transitions écologiques entre différents états.

# ## Suppositions du modèle :
# - Le nombre de parcelles initiales est de 200 et reste le même tout au long de la simulation.
# - Les parcelles sont indépendantes les unes des autres, ce qui signifie que la transition d'une parcelle d'un état à un autre ne dépend pas de l'état des parcelles voisines.
# - Il n'existe que quatre états possibles pour chaque parcelle: vide, herbes, buissons de type A et buissons de type B. 
# - Le début du modèle correspond à une situation où les arbres ont été coupés, et il n'y a donc que des parcelles vides ou occupées par des buissons.
# - Au début de la simulation, la majorité des parcelles sont vides (80%), tandis qu'une proportion plus faible est occupée par des buissons (20%).
# - L'observation de l'évolution de la communauté végétale se fait sur une période de 200 générations.

# ## Décisions principales : 
# - La population initiale est composée de 150 parcelles vides, 35 parcelles occupées par des buissons de type A et 15 parcelles occupées par des buissons de type B, pour un total de 200 parcelles.
# - L'évolution du paysage est modélisée à l'aide d'une matrice de transition. Cette matrice décrit la probabilité qu'une parcelle passe d'un état à un autre entre deux générations. Par exemple, une parcelle
# vide peut rester vide ou peut devenir une parcelle d'herbes ou de buissons. 
# - Pour les transitions végétales, les régles suivantes peuvent s'appliquer  : 
# 1. une parcelle vide peut rester vide ou être colonisée par des herbes ou des buissons
# 2. une parcelle avec des herbes peut rester herbacée ou évoluer vers un état de buisson
# 3. une parcelle occupée par un buisson peut rester stable ou redevenir vide à la suite d'une perturbation
#  - Les transitions entre les différents étants se font de manière aléatoire, mais les probabilités de transition sont définies de manière à favoriser la persistance 
# des parcelles vides tout en permettant une colonisation limitée par les herbes et les buissons, afin de respecter les contraintes de sécurité et de conservation.

# # Implémentation du modèle 
# ## Packages nécessaires :
import Random
Random.seed!(123456)
import Distributions
using CairoMakie

# ## Paramètres du modèle : 
s = [150, 0, 35, 15] # soit 150 parcelles vides, 0 parcelles d'herbes, 35 parcelles de buissons de type A et 15 parcelles de buissons de type B, pour un total de 200 parcelles.
states = length(s)
patches = sum(s)
generations = 200
nb.parcelles = 200

# ## Information pour la visualisation du graphique :
states_names = ["Vide", "Herbes", "Buissons A", "Buissons B"]
states_colors = [:grey40, :orange, :teal, :purple]


# ## Fonction de vérification :
"""
    check_transition_matrix!(T)

Cette fonction vérifie que chaque ligne de la matrice de transition somme à 1. Si ce n'est pas le cas, elle normalise la ligne en divisant chaque valeur par la somme de la ligne.
"""
function check_transition_matrix!(T)
    for ligne in axes(T, 1)
        if sum(T[ligne, :]) != 1
            @warn "La somme de la ligne $(ligne) n'est pas égale à 1 et a été modifiée"
            T[ligne, :] ./= sum(T[ligne, :])
        end
    end
    return T
end

"""
    check_function_arguments(transitions, states)

Cette fonction vérifie que les dimensions des arguments sont valides: la matrice de transition doit être carrée et le nombre d'états doit correspondre au nombre de lignes de la matrice
"""
function check_function_arguments(transitions, states)
    if size(transitions, 1) != size(transitions, 2)
        throw("Le nombre d'états ne correspond pas à la matrice de transition")
    end
    return nothing
end

# ## Fonctions de simulation :
"""
    _sim_stochastic!(timeseries, transitions, generation)

Cette fonction simule une transition stochastique entre les états en utilisant les probabilités de la matrice
"""
function _sim_stochastic!(timeseries, transitions, generation)
    for state in axes(timeseries, 1)
        pop_change = rand(Multinomial(timeseries[state, generation], transitions[state, :]))
        timeseries[:, generation+1] .+= pop_change
    end
end

"""
    _sim_deterministe!(timeseries, transitions, generation)

Cette fonction simule une transition déterministe en multipliant le vecteur d'états par la matrice de transition
"""
function _sim_deterministe!(timeseries, transitions, generation)
    pop_change = (timeseries[:, generation]' * transitions)'
    timeseries[:, generation+1] .+= pop_change
end

# ## Fonction principale de la simulation
"""
    simulation(transitions, states)

Cette fonction est celle principale qui exécute le simulation du modèle de transition sur plusieurs générations
"""
function simulation(transitions, states; generations=500, stochastic=false)
    check_transition_matrix!(transitions)
    check_function_arguments(transitions, states)

    _data_type = stochastic ? Int64 : Float32
    timeseries = zeros(_data_type, length(states), generations + 1)
    timeseries[:, 1] = states

    _sim_function! = stochastic ? _sim_stochastic! : _sim_deterministe!

    for generation in Base.OneTo(generations)
        _sim_function!(timeseries, transitions, generation)
    end
    return timeseries
end

# ## Matrice de transition
# Les valeurs représentent des probabilités relatives de transition entre états.

# Transitions parcelles vides :
# Ces valeurs vont devenir des probabilités plus tard, et donc des gros chiffres ont été utilisés afin de s'assurer qu'à l'équilibre, seulement 20% des parcelles sont végétalisées
# La somme de toutes les parcelles végétalisées est 40 (15+15+10), soit 20% du total des parcelles. Parmi ces 40 parcelles végétalisées, 70% sont des buissons (28 parcelles) et 30% sont des herbes (12 parcelles)
T = zeros(Float64, states, states)
T[1, :] = [375, 10, 9, 6]

# Transitions parcelles herbes :
# Les parcelles d'herbes restent majoritairement des herbes, mais peuvent aussi changer parmi ces 20%, 70% sont des buissons(28 parcelles) 
T[2, :] = [120, 230, 35, 15]

# Transitions parcelles buissons :
# Les parcelles de buissons de type A restent majoritairement des buissons de type A, mais peuvent aussi changer parmi ces 20%, 70% sont des buissons (28 parcelles) (un type de buisson doit être plus abondant que l'autre)
# Il y a donc 18 parcelles de Buissons_A et 10 Buissons_B
T[3, :] = [90, 15, 280, 15] #transition comme buisson_A 
T[4, :] = [95, 15, 20, 270] #transition comme buisson_B


# # Présentation des résultats
f = Figure()
ax = Axis(f[1, 1], xlabel="Nb. générations", ylabel="Nb. parcelles", yticks=0:10:200)
# ## Simulation stochastique :
for _ in 1:100
    sto_sim = simulation(T, s; stochastic=true, generations=200)
    for i in eachindex(s)
        lines!(ax, sto_sim[i, :], color=states_colors[i], alpha=0.1)
    end
end
# ## Simulation déterministe : 
det_sim = simulation(T, s; stochastic=false, generations=200)
for i in eachindex(s)
    lines!(ax, det_sim[i, :], color=states_colors[i], alpha=1,
        label=states_names[i], linewidth=4)
end

axislegend(ax)
tightlimits!(ax)
current_figure()

# # Discussion
# Les résultats obtenus avec cette simulation sont cohérents avec notre
# hypothèse initiale selon laquelle l’introduction de deux espèces de 
# buissons dans les parcelles permettrait de créer une biodiversité 
# végétale relativement stable sous la ligne électrique. Le modèle 
# démontre également qu’une proportion majoritaire des parcelles demeure
# vide, tandis qu’une proportion plus petite est occupée par de la 
# végétation basse comme des herbacées et des buissons. Cette structure
# semble donc optimale afin de concilier les besoins écologiques et les 
# contraintes de sécurité sous les corridors électriques 
# (kimmel2024integratedvegetationmanagement).

# En effet, la population initiale choisie est composée de 150 parcelles 
# vides, 35 parcelles occupées par des buissons de type A et 15 parcelles
# occupées par des buissons de type B, pour un total de 200 parcelles. La
# matrice de transition a été ajustée afin de favoriser la persistance des
# parcelles vides tout en permettant une colonisation limitée par les 
# herbes et les buissons. Cette configuration conduit le modèle vers un
# état d'équilibre d'environ 160 parcelles vides et 40 parcelles 
# végétalisées. Parmi ces parcelles végétalisées, environ 30 % sont 
# occupées par des herbes et 70 % par des buissons, ce qui respecte les
# contraintes imposées. De plus, la variété de buisson la moins abondante 
# (type B) représente plus de 30 % du total des buissons, satisfaisant 
# ainsi la condition de diversité.

# Les simulations stochastiques montrent également que, malgré la 
# variabilité associée aux transitions aléatoires entre états, les
# proportions de végétation se stabilisent autour des valeurs prévues par
# le modèle déterministe. Cela suggère que la configuration choisie pour 
# la population initiale et la matrice de transition produit un système 
# relativement robuste. En effet, même lorsque les transitions sont 
# soumises à des fluctuations aléatoires, la majorité des trajectoires 
# converge vers un équilibre similaire, indiquant que les critères de
# gestion peuvent être respectés dans plusieurs simulations. Le modèle
# déterministe, quant à lui, représente la tendance moyenne du système et
# permet de visualiser clairement l'état d'équilibre vers lequel les
# simulations stochastiques convergent (gustafsson2013whencan).

# Cependant, ce modèle comporte certaines limitations importantes. Tout 
# d’abord, les parcelles sont considérées comme indépendantes les unes des
# autres, ce qui ne reflète pas complètement les conditions écologiques 
# réelles, où la dispersion des graines (par le vent ou les animaux), la
# compétition intra- et interspécifique ainsi que les interactions entre
# parcelles voisines peuvent influencer la colonisation et la succession
# végétale. De plus, le modèle ne prend pas en compte d’autres facteurs 
# environnementaux importants, tels que le type de sol ou les 
# perturbations climatiques et anthropiques. Enfin, le modèle ne 
# représente que trois types de végétation (herbacées, buissons de type A
# et buissons de type B), alors que, dans la réalité, une plus grande 
# diversité d’espèces végétales de petite taille peut être présente sous
# les corridors électriques.

'references.bib'
@clarke2008towardsecologicalmanagement Clarke, R. H., & Johnson, M. S. (2008). Toward ecological management of power line corridors: A review of current practices and future directions. *Environmental Management*, 42(3), 345-358.
@gustafsson2013whencan Gustafsson, L., & Jonsson, B. G. (2013). When can stochasticity be ignored in population models? *Ecology Letters*, 16(7), 877-888.
@kimmel2024integratedvegetationmanagement Kimmel, K., & Smith, J. (2024). Integrated vegetation management for power line corridors: Balancing safety and biodiversity. *Journal of Environmental Management*, 300, 113456.

