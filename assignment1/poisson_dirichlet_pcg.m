function psi = poisson_dirichlet_pcg(rhs, x, y)
% Solves the finite-difference problem
%   -Laplacian(psi) = rhs
% on a rectangular grid with homogeneous Dirichlet boundary conditions
% (psi = 0 on the outer boundary).

    [Ny, Nx] = size(rhs);

    x = x(:).';
    y = y(:);

    dx = abs(median(diff(x)));
    dy = abs(median(diff(y)));

    rhs(~isfinite(rhs)) = 0;

    % Unknowns are interior points only (Dirichlet boundary on outer boundary)
    interior = false(Ny, Nx);
    interior(2:end-1, 2:end-1) = true;

    idx = find(interior);
    N = numel(idx);

    psi = zeros(Ny, Nx);
    if N == 0
        return;
    end

    % Map interior grid points to linear indices 1..N
    map = zeros(Ny, Nx);
    map(interior) = 1:N;

    cx = 1/dx^2;
    cy = 1/dy^2;

    % Build sparse matrix for -Laplacian (SPD)
    A = spalloc(N, N, 5*N);
    b = rhs(interior);

    for k = 1:N
        [i,j] = ind2sub([Ny, Nx], idx(k));

        % Diagonal: contributions from four neighbors
        A(k,k) = 2*cx + 2*cy;

        % Off-diagonals only if neighbor is also an interior unknown.
        % If neighbor is on the Dirichlet boundary, psi=0 there.
        if interior(i, j+1), A(k, map(i, j+1)) = -cx; end
        if interior(i, j-1), A(k, map(i, j-1)) = -cx; end
        if interior(i+1, j), A(k, map(i+1, j)) = -cy; end
        if interior(i-1, j), A(k, map(i-1, j)) = -cy; end
    end

    psi_int = A \ b;

    psi(interior) = psi_int;
end
