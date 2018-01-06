type InteractiveState{S} <: AbstractInteractiveState{S}
    env_state::S
    model::Model    #Other agent's model not to be confused with model of the universe
end

env_state(is::InteractiveState) =  is.env_state
model(is::InteractiveState) = is.model
