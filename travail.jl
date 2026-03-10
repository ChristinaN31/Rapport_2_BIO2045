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

# # Implémentation

# ## Packages nécessaires

import Random
Random.seed!(123456)
using CairoMakie
using Distributions

# ## Une autre section
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



    """
    foo(x, y)

Cette fonction ne fait rien.
"""
function foo(x, y)
    ## Cette ligne est un commentaire
    return nothing
end

# # Présentation des résultats

# La figure suivante représente des valeurs aléatoires:

hist(randn(100))

# # Discussion

# On peut aussi citer des références dans le document `references.bib`,
# @ermentrout1993cellular -- la bibliographie sera ajoutée automatiquement à la
# fin du document.
