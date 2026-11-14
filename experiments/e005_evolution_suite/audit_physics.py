"""Independent autograd check of scalar PDE losses for the five E005 equations."""
import torch


def mlp(theta, x, lo, hi, outputs):
    """Decode the flat affine/tanh network without the experiment's spatial jet code."""
    x = 2 * (x - lo) / (hi - lo) - 1
    width = next(
        w for w in range(1, 513)
        if (x.shape[1] + 1) * w + (w + 1) * w + (w + 1) * outputs == theta.numel()
    )
    start = 0
    for ins, outs, hidden in ((x.shape[1], width, True), (width, width, True), (width, outputs, False)):
        size = ins * outs
        weight = theta[start:start + size].reshape(outs, ins)
        bias = theta[start + size:start + size + outs]
        x = torch.nn.functional.linear(x, weight, bias)
        if hidden:
            x = x.tanh()
        start += size + outs
    return x


def pde_mse(case, theta, points, lo, hi, outputs):
    x = points.clone().detach().requires_grad_()
    value = mlp(theta.detach(), x, lo, hi, outputs)
    lap = []
    time_derivative = []
    for channel in range(outputs):
        d = torch.autograd.grad(value[:, channel].sum(), x, create_graph=True)[0]
        time_derivative.append(d[:, -1])
        lap.append(sum(
            torch.autograd.grad(d[:, j].sum(), x, retain_graph=True)[0][:, j]
            for j in range(x.shape[1] - 1)
        ))
    if case == "nls":
        z = torch.complex(value[:, 0], value[:, 1])
        zt = torch.complex(*time_derivative)
        zxx = torch.complex(*lap)
        r = 1j * zt + 0.5 * zxx + z.abs().square() * z
        residual = torch.view_as_real(r)
    elif case == "cgle":
        z = torch.complex(value[:, 0], value[:, 1])
        rhs = z + (1 + 1j) * torch.complex(*lap) - (1 + 0.5j) * z.abs().square() * z
        residual = torch.view_as_real(torch.complex(*time_derivative) - rhs)
    elif case in ("gray_scott", "fhn"):
        u, v = value.unbind(1)
        if case == "gray_scott":
            du, dv = 2e-5, 1e-5
            ru, rv = -u * v.square() + 0.04 * (1 - u), u * v.square() - 0.1 * v
        else:
            du, dv = 1e-3, 5e-3
            ru, rv = u - u.pow(3) - 0.005 - v, u - v
        residual = torch.stack((time_derivative[0] - du * lap[0] - ru,
                                time_derivative[1] - dv * lap[1] - rv), dim=1)
    elif case == "allen_cahn":
        u = value[:, 0]
        residual = time_derivative[0] - 1e-4 * lap[0] - 5 * (u - u.pow(3))
    else:
        raise ValueError(case)
    return residual.detach().square().mean().item()
