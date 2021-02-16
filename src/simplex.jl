export simplex

"""
    ... = simplex(A, b, c, IB)

Solves the linear problem

     min  cᵀx
    s.to  Ax = b
           x ≥ 0.

The starting basis `IB` must be provided.
"""
function simplex(
    A, b, c, IB;
    max_iter = 1000,
    max_time = 10.0,
    ϵ = sqrt(eps())
  )
  m, n = size(A)
  # B xB = b
  x = zeros(n)
  x[IB] = A[:,IB] \ b

  if !all(x .≥ 0)
    error("x is not positive")
  end

  iter = 0
  start_time = time()
  Δt = 0.0

  status = :unknown
  solved = false
  tired = iter ≥ max_iter > 0 || Δt ≥ max_time > 0
  unlimited = false

  while !(solved || tired || unlimited)
    y = A[:,IB]' \ c[IB]
    c̄ = c - A' * y
    @debug("Iter $iter", c̄)
    if all(c̄ .≥ -ϵ)
      solved = true
      continue
    end
    j = findfirst(c̄ .< -ϵ)
    @debug("", j)

    d = zeros(n)
    d[j] = 1
    d[IB] = A[:,IB] \ -A[:,j]
    if all(d .≥ -ϵ)
      unlimited = true
      continue
    end

    J = findall(d .< -ϵ)
    k = argmin(-x[J] ./ d[J])
    k = J[k]
    θ = -x[k] / d[k]
    @debug("", k, d, θ)

    x += θ * d
    IB = union(setdiff(IB, [k]), [j])

    iter += 1
    Δt = time() - start_time
    tired = iter ≥ max_iter > 0 || Δt ≥ max_time > 0
  end

  if solved
    status = :solved
  elseif unlimited
    status = :unlimited
  elseif tired
    if iter ≥ max_iter > 0
      status = :max_iter
    elseif Δt ≥ max_time > 0
      status = :max_time
    end
  end

  return x, status, iter, Δt
end