function D = load_piv_uv_sample(matfile, frame)
% Takes in velocity_vec.mat (USGS ScienceBase style).

if nargin < 2, frame = 1; end

S = load(matfile);

u = S.u_mean;
v = S.v_mean;

x = S.x_ground(1,:);
y = S.y_ground(:,1);

if isfield(S, "typevector_filt")
    tv = S.typevector_filt;
    k  = min(max(1, frame), size(tv,3));
    mask = (tv(:,:,k) == 1);
else
    mask = isfinite(u) & isfinite(v);
end

u = double(u); v = double(v);
x = double(x(:)'); y = double(y(:));
mask = logical(mask);

u(~mask) = NaN;
v(~mask) = NaN;

D = struct("u", u, "v", v, "x", x, "y", y, "mask", mask);
end