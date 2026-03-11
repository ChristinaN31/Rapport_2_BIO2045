# ---
# title: Titre du travail
# repository: tpoisot/BIO245-modele
# auteurs:
#    - nom: Nadler
#      prenom: Christina
#      matricule: 20313890
#      github: premierAuteur
#    - nom: Auteur
#      prenom: Deuxième
#      matricule: XXXXXXXX
#      github: DeuxiAut
# ---

# # Introduction
# hypothèse possible?? : Dans le contexte de l’aménagement d’un corridor sous une ligne électrique à haute tension, nous faisons l’hypothèse que l’introduction de deux espèces de buissons, 
# combinée à une proportion limitée d’herbes et à une forte proportion de parcelles vides, permet d’établir une communauté végétale stable qui empêche l’installation d’arbres de grande taille 
# tout en maintenant un certain niveau de biodiversité. Une telle structure végétale devrait donc permettre de concilier les contraintes de sécurité liées aux infrastructures électriques avec 
# des objectifs écologiques de conservation de la biodiversité. (on veut limiter la présence d'arbres dans ce cas parce qu'ils peuvent atteindre une hauteur importante et leurs branches peuvent
# entrer en contact avec les lignes électriques)

# # Présentation du modèle
# ## États du modèle:
# Le modèle est composé de 200 parcelles pouvant se trouver dans quatre états possibles: vide, herbes, buissons de type A et buissons de type B. Ces états représentent différents niveaux de végétation
# sous une ligne électrique, où la présence de végétation doit être contrôlée pour assurer la sécurité de l'infrastructure tout en maintenant un certain niveau de biodiversité.
# ## Population initiale
# Au début de la simulation, le modèle contient 200 parcelles. Parmi celle-ci, 150 sont vides, aucune parcelle n'est occupée par des herbes et 50 parcelles sont occupées par des buissons (35 de type A et 15 de type B). 
# Cela représente l'état initial du modèle après l'intervention humaine.
s = [150, 0, 35, 15] # rappel qu'à l'équilibre, seulement 20% des parcelles sont végétalisées
# ## Transitions écologiques
# L'évolution du paysage est modélisée à l'aide d'une matrice de transition. Cette matrice décrit la probabilité qu'une parcelle passe d'un état à un autre entre deux générations. Par exemple, une parcelle
# vide peut rester vide ou peut devenir une parcelle d'herbes ou de buissons. 

# # Implémentation
# ## Packages nécessaires 
import Random
Random.seed!(123456)
using CairoMakie
using Distributions

# ## Vérifications des entrées du modèle
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

# ## Fonctions de simulation
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

# ## Fonction principale
"""
    simulation(transitions, states)

Cette fonction est celle principale qui exécute le simulation du modèle de transition sur plusieurs générations
"""
function simulation(transitions, states; generations = 500, stochastic = false)
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

# ## Paramètres du modèle
# Il y a 4 états du paysage: vide, herbes, buisson_A, buisson_B
# Il y a 200 parcelles, toutes initiallement vides mais on peut planter jusqu'à 50 parcelles de buissons
# on va donc planter 35 parcelles de buisson_A et 15 parcelles de buisson_B
# alors il y a 150 vides, 0 herbes, 35 buissons_A et 15 buissons_B

# États initiaux
# Vide, Herbes, Buissons_A, Buissons_B
s = [150, 0, 35, 15] #note qu'à l'équilibre, on veut que seulement 20% des parcelles soient végétalisées... ça veut dire que les parcelles vides doivent rester stables et les transitions vers les parcelles végétalisées doivent être limitées  
states = length(s)
patches = sum(s)

# Information pour le graphique
states_names = ["Vide", "Herbes", "Buissons A", "Buissons B"]
states_colors = [:grey40, :orange, :teal, :purple]

# ## Matrice de transition
T = zeros(Float64, states, states)
T[1, :] = [375, 10, 9, 6] #transitions comme vide
# ces valeurs vont devenir des probabilités plus tard, et donc on a choisi d'utiliser des gros chiffres pour assurer qu'à l'équilibre, seulement 20% des parcelles sont végétalisées
# la somme de toutes les parcelles végétalisées est 40 (15+15+10)
# parmi ces 20%, 30$ doivent être des parcelles herbes
# 0.30 * 40 = 12 
T[2, :] = [120, 230, 35, 15] #transition comme herbes
# les parcelles herbes restent majoritairement herbes mais peuvent aussi changer 
# parmi ces 20%, 70% sont des buissons (un type de buisson doit être plus abdondant que l'autre)
# 0.70 * 40 = 28
# on peut dire que il y a 18 parcelles de Buissons_A et 10 Buissons_B
T[3, :] = [90, 15, 280, 15] #transition comme buisson_A 
T[4, :] = [95, 15, 20, 270] #transition comme buisson_B


#je pense qu'on peut enlever cette partie?? 
"""
    foo(x, y)

Cette fonction ne fait rien.
"""
function foo(x, y)
    ## Cette ligne est un commentaire
    return nothing
end

# # Présentation des résultats
f = Figure()
ax = Axis(f[1, 1], xlabel="Nb. générations", ylabel="Nb. parcelles", yticks = 0:10:200)
#simulation stochastique
for _ in 1:100
    sto_sim = simulation(T, s; stochastic = true, generations = 200)
    for i in eachindex(s)
        lines!(ax, sto_sim[i, :], color=states_colors[i], alpha = 0.1)
    end
end
#simulation déterministe
det_sim = simulation(T, s; stochastic = false, generations = 200)
for i in eachindex(s)
    lines!(ax, det_sim[i, :], color=states_colors[i], alpha = 1,
    label=states_names[i], linewidth=4)
end

axislegend(ax)
tightlimits!(ax)
current_figure()

# La figure suivante représente des valeurs aléatoires:
# on peut enlever cette partie?
hist(randn(100))

# # Discussion


# Les résultats de la simulation sont cohérents avec notre hypothèse initiale, qui était que l'introduction de deux espèces de buissons permettrait de créer une biodiversité végétale stable
# sous la ligne électrique tout en limitant la croissance d'arbres et en maintenant une certaine biodiversité. En réalité, les corridors sous les lignes de transport d'électricité doivent être
# maintenus libres de végétation haute afin d'éviter tout contact avec les infrastructures et de réduire les risques de panne ou d'incendie (clarke2008towardsecologicalmanagement). Toutefois, plusieurs études
# ont montré que ces zones peuvent aussi servir d'habitats importants pour de nombreuses espèces lorsqu'elles sont gérées correctement (kimmel2024integratedvegetationmanagement). Une stratégie fréquemment utilisée
# dans ces milieux est la gestion intégrée de la végétation (Integrated Vegetation Management), qui encourage la croissance de certaines plantes comme les herbes et les buissons, tout
# en empêchant la croissance d'arbres qui peuvent interférer avec les lignes électriques. Dans la pratique, la croissance des arbres est généralement contrôlée avec l'utilisation d'herbicides
# sélectifs ou le débroussaillage mécanique, qui permettent d'éliminer les espèces susceptibles de devenir trop grandes tout en laissant en place les plantes compatibles avec les contraintes du
# corridor (kimmel2024integratedvegetationmanagement). 

# Dans notre simulation, l'introduction d'une deuxième espèce de buisson permet de reproduire de ce type de type de végétation. La population initiale choisie est 
# composée de 150 parcelles vides, 35 parcelles occupées par des buissons de type A et 15 parcelles occupées par des buissons de type B, pour un total de 200 parcelles. La matrice de 
# transition a été ajustée afin de favoriser la persistance des parcelles vides tout en permettant une colonisation limitée par les herbes et les buissons. Cette configuration conduit 
# le modèle vers un état d'équilibre d'environ 160 parcelles vides et 40 parcelles végétalisées. Parmi ces parcelles végétalisées, environ 30% sont occupées par des herbes et 70% par 
# des buissons, ce qui respecte les contraintes imposées. De plus, la variété de buisson la moins abondante représente plus de 30% du total des buissons, satisfaisant ainsi la condition 
# de diversité.  Les résultats montrent donc qu'à l'équilibre, la majorité des parcelles végétalisées est occupée par des buissons, tandis que les herbes demeurent présentes mais en proportion 
# plus faible. Cette structure correspond bien aux stratégies de gestion utilisées dans les corridors de transport d'électricité, où la croissance des herbes et des buissons sont encouragés 
# car ils offrent une couverture végétale relativement basse qui n'entre pas en contact avec les infrastructures. Les simulations stochastiques montrent que, malgré la variabilité associées 
# aux transitions aléatoires entre états, les proportions de végétation se stablisent autour des valeurs prévues par le modèle déterministe. Cela suggère que la configuration choisie pour la 
# population initiale et la matrice de transition produit un système relativement solide. En effet, même lorsque les transitions sont soumises à des fluctuations aléaotoires, la majorité des
# trajectoires converge vers un équilibre similaire, indiquant que les critères de gestion peuvent être respectés dans plusieurs simulations. Par contre, le modèle déterministe représente
# la tendance moyenne du système et permet de visualiser clairement l'état d'équilibre vers lequel les simulations stochastiques convergent (gustafsson2013whencan).   

# Dans un contexte réel de gestion des corrdors électriques, ces résultats illustrent l'importance de choisir des espèces végétales dont les caractérsitiques écologiques favorisent la stabilité
# de la communauté végétale tout en limitant la croissance des espèces qu'on veut éliminer. Une communauté composée par des buissons de petites taille peut donc être une solution efficace pour
# assurer que les contraintes de sécurité des infrastructures, mais aussi les objectifs de conservation de biodiversité, sont respectés.




