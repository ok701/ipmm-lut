import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.interpolate import RegularGridInterpolator
from matplotlib.widgets import Slider, Button, TextBox

# =========================
# Shared motor model
# =========================
def Te(id_, iq, p_):
    return 1.5 * p_["pole_pairs"] * (p_["psi_f"] * iq + (p_["Ld"] - p_["Lq"]) * id_ * iq)

def lam_d(id_, p_):
    return p_["psi_f"] + p_["Ld"] * id_

def lam_q(iq, p_):
    return p_["Lq"] * iq

def lam_mag(id_, iq, p_):
    return float(np.hypot(lam_d(id_, p_), lam_q(iq, p_)))

def part1_lambda_max_ff(rpm, Vdc, p_):
    omega_mech = rpm * 2 * np.pi / 60.0
    omega_e = p_["pole_pairs"] * omega_mech
    Vmax = p_["alpha"] * Vdc
    return float(Vmax / max(abs(omega_e), 1e-9))

def solve_Tmax_for_lammax(lam_max, p_):
    Imax = p_["Imax"]
    bounds = [(-Imax, 0.0), (0.0, Imax)]

    def obj(x):
        return -Te(x[0], x[1], p_)

    cons = [
        {"type": "ineq", "fun": lambda x: Imax**2 - (x[0] ** 2 + x[1] ** 2)},
        {"type": "ineq", "fun": lambda x: lam_max - lam_mag(x[0], x[1], p_)},
    ]

    inits = []
    inits.append(np.array([0.0, min(Imax, 5.0)]))
    for frac in [0.2, 0.5, 0.8, 1.0]:
        iq0 = frac * Imax
        id0 = -np.sqrt(max(Imax**2 - iq0**2, 0.0))
        inits.append(np.array([id0, iq0]))
    inits.append(np.array([-0.5 * Imax, 0.5 * Imax]))

    best_T = -np.inf
    best_x = (np.nan, np.nan)

    for x0 in inits:
        res = minimize(
            obj, x0, method="SLSQP", bounds=bounds, constraints=cons,
            options={"maxiter": 800, "ftol": 1e-12, "disp": False}
        )
        if not res.success:
            continue

        id_, iq = res.x
        if (id_**2 + iq**2) > Imax**2 + 1e-6:
            continue
        if lam_mag(id_, iq, p_) > lam_max + 1e-6:
            continue

        val = Te(id_, iq, p_)
        if val > best_T:
            best_T = float(val)
            best_x = (float(id_), float(iq))

    if not np.isfinite(best_T):
        return np.nan, (np.nan, np.nan)
    return best_T, best_x

def build_part3_LUT(lam_max_grid, p_):
    Tmax_LUT   = np.full_like(lam_max_grid, np.nan, dtype=float)
    Id_at_Tmax = np.full_like(lam_max_grid, np.nan, dtype=float)
    Iq_at_Tmax = np.full_like(lam_max_grid, np.nan, dtype=float)

    for k, lm in enumerate(lam_max_grid):
        Tmax, (id_opt, iq_opt) = solve_Tmax_for_lammax(float(lm), p_)
        Tmax_LUT[k]   = Tmax
        Id_at_Tmax[k] = id_opt
        Iq_at_Tmax[k] = iq_opt
    return Tmax_LUT, Id_at_Tmax, Iq_at_Tmax

def solve_min_current_for_T_lam(Tref, lam_ref, p_, x0=None):
    Imax = p_["Imax"]
    bounds = [(-Imax, 0.0), (0.0, Imax)]

    def cost(x):
        return x[0] ** 2 + x[1] ** 2

    cons = [
        {"type": "eq", "fun": lambda x: Te(x[0], x[1], p_) - Tref},
        {"type": "ineq", "fun": lambda x: lam_ref - lam_mag(x[0], x[1], p_)},
        {"type": "ineq", "fun": lambda x: Imax**2 - (x[0] ** 2 + x[1] ** 2)},
    ]

    inits = []
    if x0 is not None:
        inits.append(np.array(x0, dtype=float))

    denom = 1.5 * p_["pole_pairs"] * max(p_["psi_f"], 1e-9)
    iq0 = Tref / denom
    if 0.0 <= iq0 <= Imax:
        inits.append(np.array([0.0, iq0]))

    for frac in [0.2, 0.5, 0.8, 1.0]:
        iq_guess = frac * Imax
        id_guess = -np.sqrt(max(Imax**2 - iq_guess**2, 0.0))
        inits.append(np.array([id_guess, iq_guess]))
    inits.append(np.array([-0.2 * Imax, 0.2 * Imax]))

    best_cost = np.inf
    best_x = None
    best_res = None

    for x0_try in inits:
        res = minimize(
            cost, x0_try, method="SLSQP", bounds=bounds, constraints=cons,
            options={"maxiter": 1200, "ftol": 1e-12, "disp": False}
        )
        if not res.success:
            continue
        id_, iq = res.x
        if abs(Te(id_, iq, p_) - Tref) > 1e-3:
            continue
        if lam_mag(id_, iq, p_) > lam_ref + 1e-6:
            continue
        if (id_**2 + iq**2) > Imax**2 + 1e-6:
            continue
        c = cost(res.x)
        if c < best_cost:
            best_cost = c
            best_x = res.x.copy()
            best_res = res
    return best_x, best_res

# =========================
# Global variables/State
# =========================
p_ = {
    "pole_pairs": 4,
    "Ld":         0.004,
    "Lq":         0.008,
    "psi_f":      0.01,
    "Imax":       20.0,
    "alpha":      1/3,
}
Vdc = 48.0
rpm_max = 6000
rpm_min = 0

lam_grid = None
Tratio_grid = None
Tmax_LUT = None
Id_at_Tmax = None
Iq_at_Tmax = None
interp_id = None
interp_iq = None
Id_LUT_2D = None
Iq_LUT_2D = None

def rebuild_lut():
    global lam_grid, Tratio_grid, Tmax_LUT, Id_at_Tmax, Iq_at_Tmax, interp_id, interp_iq, Id_LUT_2D, Iq_LUT_2D
    N_lam = 20
    N_tref = 20

    lam_upper = np.hypot(p_["psi_f"] + p_["Ld"] * p_["Imax"], p_["Lq"] * p_["Imax"]) * 1.05
    # Start from 2% of max flux so the LUT covers all speeds (independent of rpm_max)
    lam_lower = lam_upper * 0.02

    lam_grid = np.linspace(lam_lower, lam_upper, N_lam)
    Tratio_grid = np.linspace(0.0, 0.999, N_tref)

    Tmax_LUT, Id_at_Tmax, Iq_at_Tmax = build_part3_LUT(lam_grid, p_)

    Id_LUT_2D = np.full((N_lam, N_tref), np.nan)
    Iq_LUT_2D = np.full((N_lam, N_tref), np.nan)

    for i, lam_max in enumerate(lam_grid):
        Tmax_i = float(np.interp(lam_max, lam_grid, Tmax_LUT))
        id0_w = float(np.interp(lam_max, lam_grid, Id_at_Tmax))
        iq0_w = float(np.interp(lam_max, lam_grid, Iq_at_Tmax))
        for j, ratio in enumerate(Tratio_grid):
            Tref_ij = ratio * Tmax_i
            if ratio < 0.01:
                Id_LUT_2D[i, j] = 0.0
                Iq_LUT_2D[i, j] = 0.0
                continue
            sol_ij, _ = solve_min_current_for_T_lam(Tref_ij, lam_max, p_, x0=[id0_w, iq0_w])
            if sol_ij is not None:
                Id_LUT_2D[i, j] = sol_ij[0]
                Iq_LUT_2D[i, j] = sol_ij[1]

    interp_id = RegularGridInterpolator((lam_grid, Tratio_grid), Id_LUT_2D, method='linear', bounds_error=False, fill_value=None)
    interp_iq = RegularGridInterpolator((lam_grid, Tratio_grid), Iq_LUT_2D, method='linear', bounds_error=False, fill_value=None)

# =========================
# UI Setup
# =========================
print("Building initial LUT...")
rebuild_lut()
print("Done. Launching UI...")

from matplotlib.patches import FancyBboxPatch

fig = plt.figure(figsize=(14, 8))

# =========================
# Layout grid (figure fractions)
# Left col : x  0.010 -> 0.405   width = 0.395
# Right col: x  0.425 -> 0.990   width = 0.565
# Top row  : y  0.215 -> 0.990   height = 0.775
# Bot row  : y  0.010 -> 0.190   height = 0.180
# =========================
_BOX_STYLE = dict(boxstyle="round,pad=0.01", facecolor="#F4F6F7", edgecolor="#AAB7B8", linewidth=1.5)

def _add_box(l, b, w, h):
    fig.add_artist(FancyBboxPatch((l, b), w, h, transform=fig.transFigure,
                                  clip_on=False, zorder=0, **_BOX_STYLE))

_add_box(0.010, 0.409, 0.395, 0.581)   # params box  (left-top,    6 parts)
_add_box(0.010, 0.010, 0.395, 0.387)   # LUT box     (left-bottom, 4 parts)
_add_box(0.425, 0.215, 0.565, 0.775)   # graph box   (right-top)
_add_box(0.425, 0.010, 0.565, 0.180)   # sliders box (right-bottom)

# ------------------------------------------------------------------
# RIGHT-TOP: Main interactive plot (inside graph box)
# ------------------------------------------------------------------
ax = fig.add_axes([0.460, 0.255, 0.510, 0.710])

# ------------------------------------------------------------------
# LEFT-BOTTOM: LUT heatmaps (inside LUT box, centered, square)
# ------------------------------------------------------------------
_lut_cx  = 0.2075   # center x of left column
_lut_w   = 0.125    # heatmap width (square: 0.125*14 ~ 0.220*8 ~ 1.75in)
_lut_h   = 0.220    # heatmap height
_lut_cw  = 0.011    # colorbar width
_lut_gap = 0.018    # gap between id pair and iq pair
_lut_total = 2 * (_lut_w + _lut_cw) + _lut_gap
_lut_x0  = _lut_cx - _lut_total / 2
_lut_b   = 0.010 + (0.387 - _lut_h) / 2  # vertically centered in LUT box

ax_id_lut = fig.add_axes([_lut_x0,                                          _lut_b, _lut_w,  _lut_h])
ax_id_cb  = fig.add_axes([_lut_x0 + _lut_w + 0.003,                        _lut_b, _lut_cw, _lut_h])
ax_iq_lut = fig.add_axes([_lut_x0 + _lut_w + _lut_cw + _lut_gap,           _lut_b, _lut_w,  _lut_h])
ax_iq_cb  = fig.add_axes([_lut_x0 + 2*_lut_w + _lut_cw + _lut_gap + 0.003, _lut_b, _lut_cw, _lut_h])

# ------------------------------------------------------------------
# LEFT-TOP: Parameter TextBoxes + button (inside params box, centered)
# ------------------------------------------------------------------
_p_cx  = 0.2075    # center x of left column
_f_w   = 0.155     # TextBox field width
_f_h   = 0.033     # TextBox field height
_f_gap = 0.010     # vertical gap between fields
_b_h   = 0.045     # button height
_b_w   = 0.250     # button width
_b_gap = 0.028     # gap before button

_n_fields   = 6
_content_h  = _n_fields * _f_h + (_n_fields - 1) * _f_gap + _b_gap + _b_h
_params_cy  = (0.409 + 0.990) / 2   # center y of params box (6-part, top)
_top_y      = _params_cy + _content_h / 2

def _field_b(i):   # bottom y of i-th TextBox (0 = topmost)
    return _top_y - (i + 1) * _f_h - i * _f_gap

_f_left = _p_cx - _f_w / 2

ax_vdc  = fig.add_axes([_f_left, _field_b(0), _f_w, _f_h])
ax_imax = fig.add_axes([_f_left, _field_b(1), _f_w, _f_h])
ax_psif = fig.add_axes([_f_left, _field_b(2), _f_w, _f_h])
ax_ld   = fig.add_axes([_f_left, _field_b(3), _f_w, _f_h])
ax_lq   = fig.add_axes([_f_left, _field_b(4), _f_w, _f_h])
ax_pp   = fig.add_axes([_f_left, _field_b(5), _f_w, _f_h])

t_vdc  = TextBox(ax_vdc,  'Vdc [V]',    initial=str(Vdc))
t_imax = TextBox(ax_imax, 'Imax [A]',   initial=str(p_["Imax"]))
t_psif = TextBox(ax_psif, 'psi_f [Wb]', initial=str(p_["psi_f"]))
t_ld   = TextBox(ax_ld,   'Ld [H]',     initial=str(p_["Ld"]))
t_lq   = TextBox(ax_lq,   'Lq [H]',     initial=str(p_["Lq"]))
t_pp   = TextBox(ax_pp,   'Pole Pairs',  initial=str(p_["pole_pairs"]))

_btn_b    = _field_b(5) - _b_gap - _b_h
ax_button = fig.add_axes([_p_cx - _b_w / 2, _btn_b, _b_w, _b_h])
btn_recalc = Button(ax_button, 'Update Params & Rebuild LUT')

# ------------------------------------------------------------------
# RIGHT-BOTTOM: Sliders (inside sliders box, centered)
# ------------------------------------------------------------------
_s_cx  = 0.425 + 0.565 / 2
_s_w   = 0.440
_s_h   = 0.030
_s_gap = 0.022
_sliders_cy = (0.010 + 0.190) / 2
_s_top_b    = _sliders_cy + _s_gap / 2
_s_bot_b    = _sliders_cy - _s_gap / 2 - _s_h
_s_left     = _s_cx - _s_w / 2

ax_rpm  = fig.add_axes([_s_left, _s_top_b, _s_w, _s_h])
ax_tref = fig.add_axes([_s_left, _s_bot_b, _s_w, _s_h])
s_rpm  = Slider(ax_rpm,  'rpm',       100, rpm_max, valinit=3000, valstep=50)
s_tref = Slider(ax_tref, 'Tref [Nm]', 0,   10,      valinit=2,    valstep=0.5)


dyn_artists = []
bg_artists = []
im_id = None
im_iq = None
cb_id = None
cb_iq = None
ID_bg = None
IQ_bg = None
LAM_bg = None
TE_bg = None
_ax_bg = None  # cached background for blitting (static content)

def _save_ax_bg():
    """Full-render then cache the figure pixel buffer (static content only)."""
    global _ax_bg
    fig.canvas.draw()
    _ax_bg = fig.canvas.copy_from_bbox(fig.bbox)

def init_heatmaps():
    global im_id, im_iq, cb_id, cb_iq
    ax_id_lut.clear()
    ax_iq_lut.clear()
    ax_id_cb.clear()
    ax_iq_cb.clear()

    if Id_LUT_2D is not None and Iq_LUT_2D is not None:
        im_id = ax_id_lut.imshow(Id_LUT_2D, aspect='auto', origin='lower',
                                 extent=[Tratio_grid[0], Tratio_grid[-1], lam_grid[0], lam_grid[-1]])
        ax_id_lut.set_title("Id LUT [A]", fontsize=10)
        ax_id_lut.set_xlabel("T_ratio", fontsize=8)
        ax_id_lut.set_ylabel("lam_max [Wb]", fontsize=8)
        ax_id_lut.tick_params(axis='both', which='major', labelsize=8)
        cb_id = fig.colorbar(im_id, cax=ax_id_cb)
        cb_id.ax.tick_params(labelsize=8)

        im_iq = ax_iq_lut.imshow(Iq_LUT_2D, aspect='auto', origin='lower',
                                 extent=[Tratio_grid[0], Tratio_grid[-1], lam_grid[0], lam_grid[-1]])
        ax_iq_lut.set_title("Iq LUT [A]", fontsize=10)
        ax_iq_lut.set_xlabel("T_ratio", fontsize=8)
        ax_iq_lut.tick_params(axis='both', which='major', labelsize=8)
        cb_iq = fig.colorbar(im_iq, cax=ax_iq_cb)
        cb_iq.ax.tick_params(labelsize=8)

def init_bg():
    global bg_artists, ID_bg, IQ_bg, LAM_bg, TE_bg
    for a in bg_artists:
        try:
            a.remove()
        except:
            pass
    bg_artists.clear()
    
    id_vec_bg = np.linspace(-p_["Imax"] * 1.1, p_["Imax"] * 0.1, 30)
    iq_vec_bg = np.linspace(0, p_["Imax"] * 1.1, 30)
    ID_bg, IQ_bg = np.meshgrid(id_vec_bg, iq_vec_bg)
    I_bg = np.sqrt(ID_bg**2 + IQ_bg**2)
    
    LAM_d_bg = p_["psi_f"] + p_["Ld"] * ID_bg
    LAM_q_bg = p_["Lq"] * IQ_bg
    LAM_bg = np.sqrt(LAM_d_bg**2 + LAM_q_bg**2)
    TE_bg = 1.5 * p_["pole_pairs"] * (p_["psi_f"] * IQ_bg + (p_["Ld"] - p_["Lq"]) * ID_bg * IQ_bg)
    
    # MTPA trajectory closed-form
    delta = p_["Lq"] - p_["Ld"]
    I_sweep = np.linspace(0.0, p_["Imax"], 50)
    if abs(delta) < 1e-12:
        id_mtpa, iq_mtpa = np.zeros_like(I_sweep), I_sweep.copy()
    else:
        k = p_["psi_f"] / delta
        id_mtpa = (k - np.sqrt(k**2 + 8.0 * I_sweep**2)) / 4.0
        iq_mtpa = np.sqrt(np.maximum(I_sweep**2 - id_mtpa**2, 0.0))
    v_mtpa = np.isfinite(id_mtpa) & np.isfinite(iq_mtpa) & (id_mtpa <= 1e-9) & (iq_mtpa >= -1e-9)
    
    cf = ax.contourf(ID_bg, IQ_bg, I_bg, levels=[0, p_["Imax"]], colors=['#AED6F1'], alpha=0.3)
    ct = ax.contour( ID_bg, IQ_bg, I_bg, levels=[p_["Imax"]],    colors=['#2E86C1'], linewidths=1.5)
    line_mtpa, = ax.plot(id_mtpa[v_mtpa], iq_mtpa[v_mtpa], 'orange', lw=1.8, label="MTPA")
    line_traj, = ax.plot(Id_at_Tmax, Iq_at_Tmax, 'purple', lw=1.8, label="Optimal traj.")
    
    ax.set_xlabel(r"$i_d$ [A]")
    ax.set_ylabel(r"$i_q$ [A]")
    ax.set_xlim(id_vec_bg[0], id_vec_bg[-1])
    ax.set_ylim(0, iq_vec_bg[-1])
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    bg_artists.extend([cf, ct, line_mtpa, line_traj])

point_op, = ax.plot([], [], 'ro', ms=12, mec='darkred', mew=2, label='OP')

def _update_legend():
    handles, labels = ax.get_legend_handles_labels()
    valid_handles = [h for h in handles if not h.get_label().startswith('_')]
    if valid_handles:
        ax.legend(loc="upper right", fontsize=8, handles=valid_handles)

def redraw(val=None):
    rpm = s_rpm.val
    Tref_cmd = s_tref.val
    lam_ref = part1_lambda_max_ff(rpm, Vdc, p_)
    Tmax_at_lam = float(np.interp(lam_ref, lam_grid, Tmax_LUT))
    Tref = float(np.clip(Tref_cmd, 0.0, Tmax_at_lam * 0.999))
    T_ratio = Tref / max(Tmax_at_lam, 1e-6)
    
    pt = np.array([[lam_ref, T_ratio]])
    id_op = interp_id(pt).item()
    iq_op = interp_iq(pt).item()
    
    # --- Remove previous dynamic artists ---
    for a in dyn_artists:
        try:
            a.remove()
        except:
            pass
    dyn_artists.clear()

    # --- Use globally defined ID_bg, IQ_bg, LAM_bg, TE_bg (computed once in init_bg) ---

    cf = ax.contourf(ID_bg, IQ_bg, LAM_bg, levels=[0, lam_ref], colors=['#A9DFBF'], alpha=0.4)
    ct = ax.contour( ID_bg, IQ_bg, LAM_bg, levels=[lam_ref],    colors=['#1E8449'], linewidths=1.5)
    dyn_artists.extend([cf, ct])

    if Tref_cmd > 0:
        c1 = ax.contour(ID_bg, IQ_bg, TE_bg, levels=[Tref_cmd],
                        colors=['red'], linewidths=1.5, linestyles='--')
        dyn_artists.append(c1)
    if Tref > 0:
        c2 = ax.contour(ID_bg, IQ_bg, TE_bg, levels=[Tref],
                        colors=['red'], linewidths=2.5)
        dyn_artists.append(c2)
        
    point_op.set_data([id_op], [iq_op])
    point_op.set_label("OP")
    
    ax.set_title(
        f"$i_d^*$ = $\\bf{{{id_op:.1f}}}$ A,  $i_q^*$ = $\\bf{{{iq_op:.1f}}}$ A\n"
        f"|  $T_{{ref}}$ = $\\bf{{{Tref:.1f}}}$ / {Tmax_at_lam:.1f} Nm",
        fontsize=12
    )

    _update_legend()

    # --- Blit: restore cached static background, draw only dynamic artists ---
    # In matplotlib 3.8+ ContourSet IS an Artist — draw_artist(cs) works directly.
    if _ax_bg is not None:
        try:
            fig.canvas.restore_region(_ax_bg)
            for cs in dyn_artists:
                ax.draw_artist(cs)
            ax.draw_artist(point_op)
            ax.draw_artist(ax.title)
            leg = ax.get_legend()
            if leg is not None:
                ax.draw_artist(leg)
            fig.canvas.blit(fig.bbox)
            return
        except Exception:
            pass  # fall back to full redraw on any renderer issue
    fig.canvas.draw_idle()

def on_recalc_clicked(event):
    global Vdc, rpm_max
    
    try:
        Vdc = float(t_vdc.text)
        p_["Imax"] = float(t_imax.text)
        p_["psi_f"] = float(t_psif.text)
        p_["Ld"] = float(t_ld.text)
        p_["Lq"] = float(t_lq.text)
        p_["pole_pairs"] = int(t_pp.text)
    except ValueError:
        print("Invalid input parameter! Please enter numbers.")
        btn_recalc.label.set_text('Invalid Input')
        fig.canvas.draw_idle()
        return

    btn_recalc.label.set_text('Rebuilding...')
    fig.canvas.draw_idle()
    plt.pause(0.01)
    
    rebuild_lut()
    init_heatmaps()
    init_bg()
    _save_ax_bg()

    # rpm slider range is fixed; no slider update needed

    redraw()
    btn_recalc.label.set_text('Update Core Params\n& Rebuild LUT')
    fig.canvas.draw_idle()

btn_recalc.on_clicked(on_recalc_clicked)
s_rpm.on_changed(redraw)
s_tref.on_changed(redraw)
init_heatmaps()
init_bg()
_save_ax_bg()
redraw()
plt.show()
