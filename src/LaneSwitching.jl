type CarAction <: Action
  ddot_x::Float64
  dot_y::Float64
end
==(a1::CarAction, a2::CarAction) = (a1.ddot_x == a2.ddot_x) && (a1.dot_y == a2.dot_y)
Base.hash(a1::CarAction, h::UInt64=zero(UInt64)) = hash(a1.ddot_x, hash(a1.dot_y,h))
Base.copy(a1::CarAction) = CarAction(a1.ddot_x, a1.dot_y)

type EgoActionSpace
  actions::Array{CarAction,1}
end
#I am not sure how these function are supposed to work but just adding this line  for good measure
EgoActionSpace() = EgoActionSpace([CarAction(0.0,0.0), CarAction(0.0,-2.0), CarAction(0.0,2.0), CarAction(-2.0,-2.0), CarAction(-2.0,0.0), CarAction(-2.0,2.0), CarAction(2.0,-2.0), CarAction(2.0,0.0), CarAction(2.0,2.0), CarAction(-6.0,0.0)])

Base.length(asp::EgoActionSpace) = length(asp.actions)
iterator(actSpace::EgoActionSpace) = 1:length(actSpace.actions)
dimensions(::EgoActionSpace) = 1

#Sample random action
Base.rand(rng::AbstractRNG, asp::EgoActionSpace) = Base.rand(rng, 1:Base.length(asp))

immutable CarPhysicalState <: CarState
  #absent::Bool
  state::NTuple{3,Float64} #<x, y, \dot{x} >.
end
#CarPhysicalState(state::NTuple{3,Float64}) = CarPhysicalState(false, state)
CarPhysicalState(st::CarPhysicalState) = CarPhysicalState(st.state)
==(s1::CarPhysicalState, s2::CarPhysicalState) = (s1.state == s2.state)
>(s1::CarPhysicalState, s2::CarPhysicalState) = (s1.state[1] > s2.state[2])
<(s1::CarPhysicalState, s2::CarPhysicalState) = (s1.state[1] < s2.state[2])
Base.hash(s::CarPhysicalState, h::UInt64=zero(UInt64)) = hash(s.state,h)
Base.copy(s::CarPhysicalState) = CarPhysicalState(s.state)

collision(s1::CarPhysicalState, s2::CarPhysicalState, safetyDist::Float64=0.0) = (abs(s1.state[1] - s2.state[1]) < CAR_LENGTH + safetyDist) && (abs(s1.state[2] - s2.state[2]) < CAR_WIDTH + 0.125 * safetyDist)
