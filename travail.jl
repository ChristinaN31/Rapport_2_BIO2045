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
T[1, :] = [360, 15, 15, 10] #transitions comme vide
# ces valeurs vont devenir des probabilités plus tard, et donc on a choisi d'utiliser des gros chiffres pour assurer qu'à l'équilibre, seulement 20% des parcelles sont végétalisées
# la somme de toutes les parcelles végétalisées est 40 (15+15+10)
# parmi ces 20%, 30$ doivent être des parcelles herbes
# 0.30 * 40 = 12 
T[2, :] = [80, 280, 25, 15] #transition comme herbes
# les parcelles herbes restent majoritairement herbes mais peuvent aussi changer 
# parmi ces 20%, 70% sont des buissons (un type de buisson doit être plus abdondant que l'autre)
# 0.70 * 40 = 28
# on peut dire que il y a 18 parcelles de Buissons_A et 10 Buissons_B
T[3, :] = [40, 15, 320, 25] #transition comme buisson_A 
T[4, :] = [40, 15, 25, 320] #transition comme buisson_B


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
ax = Axis(f[1, 1], xlabel="Nb. générations", ylabel="Nb. parcelles")
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

# On peut aussi citer des références dans le document `references.bib`,
# @ermentrout1993cellular -- la bibliographie sera ajoutée automatiquement à la
# fin du document.
