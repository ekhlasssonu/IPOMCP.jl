type PedestrianState_2d
    x::Float64
    y::Float64
end
==(a::PedestrianState_2d, b::PedestrianState_2d) = a.x==b.x && a.y == b.y
function Base.hash(s::PedestrianState_2d, h::UInt64=zero(UInt64))
    #print("*")
    return hash((s.x, s.y),h)
end

function print(s::PedestrianState_2d)
    print("(",s.x,",",s.y,")")
end

type CarAction2d
    accl::Float64
    ang_vel::Float64
end
==(a::CarAction2d, b::CarAction2d) = a.accl==b.accl && a.ang_vel==b.ang_vel
Base.hash(x::CarAction2d, h::UInt64=zero(UInt64)) = hash((x.accl,x.ang_vel),h)
function print(a::CarAction2d)
    print("(",a.accl,",",a.ang_vel,")")
end
