function POMCPTree(ipomdp::IPOMDP_2, sz::Int=1000)
    acts = collect(actions(ipomdp))
    A = action_type(ipomdp)
    O = obs_type(ipomdp)
    sz = min(100_000, sz)
    return BasicPOMCP.POMCPTree{A,O}(sizehint!(Int[0], sz),
                          sizehint!(Vector{Int}[collect(1:length(acts))], sz),
                          sizehint!(Array{O}(1), sz),

                          sizehint!(Dict{Tuple{Int,O},Int}(), sz),

                          sizehint!(zeros(Int, length(acts)), sz),
                          sizehint!(zeros(Float64, length(acts)), sz),
                          sizehint!(acts, sz)
                         )
end

function insert_obs_node!(t::BasicPOMCP.POMCPTree, ipomdp::IPOMDP_2, ha::Int, o)
    push!(t.total_n, 0)												# new node has been visited 0 times
    push!(t.children, sizehint!(Int[], n_actions(ipomdp)))			# for children of the new node to be inserted
    push!(t.o_labels, o)
    hao = length(t.total_n)											# index for the new node being inserted
    t.o_lookup[(ha, o)] = hao										# look up for node corresponding o from ha
    for a in actions(ipomdp)
        n = BasicPOMCP.insert_action_node!(t, hao, a)
        push!(t.children[hao], n)
    end
    return hao
end
