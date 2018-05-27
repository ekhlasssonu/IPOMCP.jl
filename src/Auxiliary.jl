function Base.rand{A}(x::Dict{A,Float64}, rng::AbstractRNG = Base.GLOBAL_RNG)
    rnd = rand(rng)
    norm_factor = 0.0
    for (k,v) in x
        norm_factor += v
    end
    sum = 0.0
    for (k,v) in x
        sum += (v/norm_factor)
        if sum >= rnd
            return k
        end
    end
    return 0.0
end
function Base.rand{A}(x::Vector{Tuple{A,Float64}}, rng::AbstractRNG = Base.GLOBAL_RNG)
    rnd = rand(rng)
    sum = 0.0
    for t in x
        sum += t[2]
        if sum >= rnd
            return t[1]
        end
    end
    return x[length(x)][1]
end

function quantal_response_probability{A}(val::Dict{A,Float64}, lambda::Float64=Inf)
    acts = collect(keys(val))
    min_value = 0.0
    for v in values(val)
        if v < min_value
            min_value = v
        end
    end

    exp_value = zeros(Float64, length(acts))
    sum = 0.0
    i = 1
    for v in values(val)
        exp_value[i] = exp(lambda * (v-min_value))
        sum += exp_value[i]
        i += 1
    end
    #println("sum = $sum")
    actProb = Dict{A,Float64}()
    if sum != Inf && sum != -Inf
        j = 1
        for act in acts
            actProb[act] = exp_value[j]/sum
            actProb[act] < 1e-5 ? actProb[act] =0.0 : nothing
            j += 1
        end
    else
        opt_actions = Vector{A}()
        opt_value = -Inf
        for i in 1:length(acts)
            if val[acts[i]] > opt_value
                empty!(opt_actions)
                opt_value = val[acts[i]]
                push!(opt_actions, acts[i])
            elseif val[acts[i]] == opt_value
                push!(opt_actions, acts[i])
            end
        end
        for act in acts
            if act in opt_actions
                actProb[act] = 1.0/length(opt_actions)
            else
                actProb[act] = 0.0
            end
        end

    end
    return actProb
end
