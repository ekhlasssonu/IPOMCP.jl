type InteractiveState{S} <: AbstractInteractiveState{S}
    env_state::S
    model::Model    #Other agent's model not to be confused with POMDP datatype as this is more general
end

==(a::InteractiveState,b::InteractiveState) = a.env_state == b.env_state && a.model == b.model
Base.hash(a::InteractiveState) = hash((a.env_state, a.model))

env_state(is::InteractiveState) =  is.env_state
model(is::InteractiveState) = is.model

function print(is::InteractiveState, numTabs::Int64=0)
    for i in 1:numTabs
        print("\t")
    end
    println("State:",is.env_state)
    #println(typeof(model))
    print(is.model, numTabs+1)
end
