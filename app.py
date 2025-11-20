import numpy as np
import matplotlib.pyplot as plt
import traceback
import matplotlib
import mpld3
import uuid
from flask import Flask, render_template, request, send_file, redirect, url_for, flash, jsonify
from scipy.signal import hilbert
import random
from scipy import signal, ndimage
import control as ct
import sympy as sp
import io, base64
from lcapy import LTIFilter
from lcapy.ltifilter import Butterworth, Bessel
import cv2
import os
from io import BytesIO
from lcapy import phasor, symbol, voltage, noisevoltage
from lcapy import StateSpace
from sympy import sympify, Matrix, simplify
import control as ct
from lcapy import texpr, s
import matplotlib.patches as patches
import skrf as rf
import math
from lcapy import symbol, nexpr, cos, sin, exp, j, sinc, rect, sincu, delta, sign, H, pi, n, f, k, dt
import html
import requests
from werkzeug.utils import secure_filename


app = Flask(__name__)  # Since your HTML is in "main/"

@app.route('/')
def index():
    return render_template('homepage.html')

@app.route('/pulse_code_mod_and_demod', methods=['GET', 'POST'])
def pcm():
    result_text = ""
    error = None
    plot_msg = ""
    plot_quant = ""
    plot_recon = ""

    if request.method == 'POST':
        try:
            # Read inputs
            fs = float(request.form.get('sampling_frequency', 100))
            fb = float(request.form.get('message_frequency', 25))
            A = float(request.form.get('amplitude', 1))
            bits_per_sample = int(request.form.get('bits', 3))
            duration = float(request.form.get('duration', 2))

            # Time vector
            t = np.arange(0, duration, 1/fs)

            # Message signal
            message = A * np.sin(2 * np.pi * fb * t)

            # Quantization
            q_levels = 2 ** bits_per_sample
            q_min = np.min(message)
            q_max = np.max(message)
            q_step = (q_max - q_min) / (q_levels - 1)
            quantized = np.round((message - q_min) / q_step) * q_step + q_min

            # Encoding
            bit_stream = []
            for q_val in quantized:
                idx = int((q_val - q_min) / q_step)
                bit_stream.append(format(idx, f'0{bits_per_sample}b'))

            # Decoding
            decoded = []
            for bits in bit_stream:
                idx = int(bits, 2)
                decoded.append(idx * q_step + q_min)
            decoded = np.array(decoded)

            def fmt_array(arr, decimals=4):
                return np.array2string(arr, formatter={'float_kind':lambda x: f"{x:.{decimals}f}"})

            # ---- Plot 1: Original Message ----
            fig1, ax1 = plt.subplots()
            ax1.plot(t, message)
            ax1.set_title("Original Message Signal")
            ax1.set_xlabel("Time (s)")
            ax1.set_ylabel("Amplitude")
            plot_msg = mpld3.fig_to_html(fig1)
            plt.close(fig1)

            # ---- Plot 2: Quantized Signal ----
            fig2, ax2 = plt.subplots()
            ax2.step(t, quantized, where='mid')
            ax2.set_title("Quantized Signal (PCM Levels)")
            ax2.set_xlabel("Time (s)")
            ax2.set_ylabel("Amplitude")
            plot_quant = mpld3.fig_to_html(fig2)
            plt.close(fig2)

            # ---- Plot 3: Reconstructed Signal ----
            fig3, ax3 = plt.subplots()
            ax3.plot(t, decoded, linestyle='dashed')
            ax3.set_title("Reconstructed Signal (After Decoding)")
            ax3.set_xlabel("Time (s)")
            ax3.set_ylabel("Amplitude")
            plot_recon = mpld3.fig_to_html(fig3)
            plt.close(fig3)

            # ---- Text Output ----
            result_text = (
                f"Sampling Frequency: {fs} Hz\n"
                f"Message Frequency: {fb} Hz\n"
                f"Amplitude: {A}\n"
                f"Quantization Levels: {q_levels}\n"
                f"Bits per Sample: {bits_per_sample}\n\n"
                 f"First 20 Quantized Samples: {fmt_array(quantized[:20])}\n\n"
                f"First 20 Encoded Bits: {bit_stream[:20]}\n\n"
                f"First 20 Decoded Samples: {fmt_array(decoded[:20])}\n"
            )

        except Exception as e:
            error = str(e)

    return render_template(
        'pulse_code_mod_and_demod.html',
        result_text=result_text, 
        error=error,
        plot_msg=plot_msg,
        plot_quant=plot_quant,
        plot_recon=plot_recon
    )

##################################################################################
@app.route('/amplitude_mod_and_demod', methods=['GET', 'POST'])
def amplitude_mod_and_demod():
    plot_html = ""
    error = None

    # Defaults
    defaults = {
        "fs": 4096e6,
        "fb": 64e6,
        "A": 2.0,
        "N_fft": 2048,
        "fc": 900e6,
        "mode": "DSBSC",
        "mu": 0.5,
        "side": "USB"
    }

    try:
        # Read inputs (POST or defaults)
        if request.method == 'POST':
            fs = float(request.form.get("fs", defaults["fs"]))
            fb = float(request.form.get("fb", defaults["fb"]))
            A = float(request.form.get("A", defaults["A"]))
            N_fft = int(request.form.get("N_fft", defaults["N_fft"]))
            fc = float(request.form.get("fc", defaults["fc"]))
            mode = request.form.get("mode", defaults["mode"])
            mu = float(request.form.get("mu", defaults["mu"]))
            side = request.form.get("side", defaults["side"])  # "USB" or "LSB"
        else:
            fs, fb, A, N_fft, fc, mode, mu, side = defaults.values()

        # Time and frequency axes
        t = np.arange(N_fft) / fs
        freqs = np.fft.fftfreq(N_fft, 1/fs)

        figs = []

        # === Baseband message (cosine) ===
        g = A * np.cos(2*np.pi*fb*t)

        # === Carrier ===
        carrier_cos = np.cos(2*np.pi*fc*t)
        carrier_sin = np.sin(2*np.pi*fc*t)

        # === Prepare analytic / Hilbert for SSB ===
        # Hilbert of g (scipy.signal.hilbert returns analytic signal)
        analytic = signal.hilbert(g)
        g_hil = np.imag(analytic)  # Hilbert transform (90 deg phase-shifted)

        # ---------------- MODULATION ----------------
        if mode == "DSBSC":
            # DSB-SC: suppressed carrier
            s = g * carrier_cos

        elif mode == "DSBAC":
            # DSB-AC (standard AM): (1 + mu * m(t)) * cos(2πfct)
            # Normalize message m(t) to unit peak to make mu meaningful.
            m_peak = np.max(np.abs(g)) if np.max(np.abs(g)) != 0 else 1.0
            m_norm = g / m_peak
            s = (1 + mu * m_norm) * carrier_cos

        elif mode == "SSBSC":
            # SSB-SC: suppressed carrier using Hilbert transform
            # SSB (USB): s = g*cos(ωc t) - g_hil * sin(ωc t)
            # SSB (LSB): s = g*cos(ωc t) + g_hil * sin(ωc t)
            if side == "USB":
                s = g * carrier_cos - g_hil * carrier_sin
            else:  # LSB
                s = g * carrier_cos + g_hil * carrier_sin

        elif mode == "SSBAC":
            # SSB with carrier (carrier added to SSB suppressed form)
            m_peak = np.max(np.abs(g)) if np.max(np.abs(g)) != 0 else 1.0
            m_norm = g / m_peak
            # suppressed SSB part
            if side == "USB":
                ssb_part = g * carrier_cos - g_hil * carrier_sin
            else:
                ssb_part = g * carrier_cos + g_hil * carrier_sin
            # add carrier scaled by (1 + mu*dc) approach -> here we add plain carrier multiplied by (1)
            # We'll use conventional AM-like carrier addition: carrier scaled by 1 (so envelope center)
            s = (1 * carrier_cos) + (mu * ssb_part)  # mu controls strength of mod content relative to carrier

        else:
            raise ValueError("Unknown mode selected")

        # ---------------- DEMODULATION (coherent) ----------------
        # Multiply received by carrier cosine (coherent demod)
        x = s * carrier_cos

        # Low-pass FIR filter design (simple windowed sinc)
        # Cutoff chosen to pass baseband content (slightly above fb)
        # normalized cutoff = fc_cut / (fs/2) for design with firwin; but we'll use sinc approach similar to original code
        # define fractional cutoff relative to sampling fs (Hz->fraction of fs)
        f_cutoff_hz = max(2*fb, fb*3)  # choose a cutoff comfortably above baseband frequency
        f_cutoff = f_cutoff_hz / fs  # fraction of fs
        # Keep f_cutoff sensible
        if f_cutoff <= 0 or f_cutoff >= 0.5:
            f_cutoff = min(max( (fb*4)/fs, 1e-6), 0.49)

        b = 0.08
        N = int(np.ceil(4 / b))
        if N % 2 == 0: N += 1
        n = np.arange(N)
        h = np.sinc(2 * f_cutoff * (n - (N - 1) / 2.0))
        w = np.blackman(N)
        h *= w
        h /= np.sum(h)

        # Convolve and keep same length
        u = np.convolve(x, h, mode='same')

        # For SSB AC/SC, envelope recovery might be different; but coherent demod + LPF recovers message for coherent receiver.

        # ---------- Plot helper ----------
        def make_plot(time_sig, freq_sig, title_time, title_freq):
            fig, axs = plt.subplots(1, 2, figsize=(9, 3))
            axs[0].plot(time_sig[:200])
            axs[0].set_title(title_time)
            axs[0].set_xlabel("Samples")
            axs[0].set_ylabel("Amplitude")

            axs[1].plot(freqs[:N_fft//2]/1e6, np.abs(freq_sig[:N_fft//2]))
            axs[1].set_title(title_freq)
            axs[1].set_xlabel("Frequency (MHz)")
            axs[1].set_ylabel("Magnitude")
            fig.tight_layout()
            return fig

        # Compute FFTs (zero-pad/truncate as needed)
        def compute_fft(sig):
            # If sig length differs from N_fft, pad or truncate for display FFT
            if len(sig) < N_fft:
                s_pad = np.zeros(N_fft)
                s_pad[:len(sig)] = sig
            else:
                s_pad = sig[:N_fft]
            return np.fft.fft(s_pad, N_fft)

        figs.append(make_plot(g, compute_fft(g), "Baseband (time)", "Baseband FFT"))
        figs.append(make_plot(carrier_cos, compute_fft(carrier_cos), "Carrier (time)", "Carrier FFT"))
        figs.append(make_plot(s, compute_fft(s), f"{mode} Modulated (time)", f"{mode} FFT"))
        figs.append(make_plot(x, compute_fft(x), "After Mixer (time)", "FFT After Mixer"))
        figs.append(make_plot(u, compute_fft(u), "Recovered (LPF output)", "FFT Recovered"))

        # Convert figures to HTML
        for f in figs:
            plot_html += mpld3.fig_to_html(f) + "<br><br>"

    except Exception as e:
        error = str(e)

    return render_template('amplitude_mod_and_demod.html',
                           plot_html=plot_html, error=error)
########################################################################################
@app.route('/frequency_mod_and_demod', methods=['GET', 'POST'])
def frequency_mod_and_demod():
    plot_html = ""
    error = None

    defaults = {
        "fs": 10000,     # Sampling frequency
        "duration": 0.01, # Short duration for clear plotting
        "fc": 500,       # Carrier frequency
        "kf": 100,       # Frequency sensitivity
        "fm": 50,        # Message frequency
        "Am": 1          # Message amplitude
    }

    try:
        if request.method == 'POST':
            fs = int(request.form.get("fs", defaults["fs"]))
            duration = float(request.form.get("duration", defaults["duration"]))
            fc = float(request.form.get("fc", defaults["fc"]))
            kf = float(request.form.get("kf", defaults["kf"]))
            fm = float(request.form.get("fm", defaults["fm"]))
            Am = float(request.form.get("Am", defaults["Am"]))
        else:
            fs, duration, fc, kf, fm, Am = defaults.values()

        # Time vector
        t = np.arange(0, duration, 1/fs)

        figs = []

        # 1. Message signal
        message = Am * np.sin(2 * np.pi * fm * t)
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.plot(t[:500], message[:500])
        ax.set_title("Message Signal (time domain)")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude")
        fig.tight_layout()
        figs.append(fig)

        # 2. FM Modulated signal
        integrated_message = np.cumsum(message) / fs
        fm_signal = np.cos(2 * np.pi * fc * t + 2 * np.pi * kf * integrated_message)
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.plot(t[:500], fm_signal[:500])
        ax.set_title("FM Modulated Signal (time domain)")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude")
        fig.tight_layout()
        figs.append(fig)

        # 3. FM Demodulated signal
        analytic_signal = hilbert(fm_signal)
        instantaneous_phase = np.unwrap(np.angle(analytic_signal))
        demodulated = np.diff(instantaneous_phase) * fs / (2 * np.pi * kf)
        demodulated = np.append(demodulated, demodulated[-1])
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.plot(t[:500], demodulated[:500])
        ax.set_title("FM Demodulated Signal (time domain)")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude")
        fig.tight_layout()
        figs.append(fig)

        # Convert all figs to HTML
        for f in figs:
            plot_html += mpld3.fig_to_html(f) + "<br><br>"

    except Exception as e:
        error = str(e)

    return render_template('frequency_mod_and_demod.html',
                           plot_html=plot_html, error=error)
###########################################################################################
@app.route('/rootlocus', methods=['GET', 'POST'])
def rootlocus():
    plot_html = ""
    routh_html = ""
    error = None

    # defaults
    defaults = {
        "num": "1",
        "den": "1 3 10 24 48 96 128 192 0",
        "K_min": 0.0,
        "K_max": 200.0,
        "K_steps": 500,
        "K_view": 128.0
    }

    def parse_poly(s):
        """Parse user text into a trimmed numpy array of floats (highest-first)."""
        if s is None:
            return np.array([0.], dtype=float)
        s = str(s).replace(',', ' ').strip()
        if not s:
            return np.array([0.], dtype=float)
        try:
            coeffs = [float(x) for x in s.split()]
        except Exception:
            # if parse fails, return zero polynomial
            return np.array([0.], dtype=float)
        arr = np.asarray(coeffs, dtype=float)
        arr = np.trim_zeros(arr, 'f')
        if arr.size == 0:
            return np.array([0.], dtype=float)
        return arr

    # robust roots helper
    def safe_roots(p):
        """
        Ensure p is numpy array, trimmed, and compute roots.
        Returns a complex numpy array (maybe empty).
        """
        p = np.asarray(p, dtype=float)
        p = np.trim_zeros(p, 'f')
        if p.size <= 1:
            # constant or empty polynomial => no roots
            return np.array([], dtype=complex)
        return np.roots(p).astype(complex)

    # numeric pole classifier (authoritative for counts)
    def classify_poles(roots, tol=1e-7):
        roots = np.asarray(roots, dtype=complex)
        if roots.size == 0:
            return 0, 0, 0, "No poles found."
        r = np.real(roots)
        rhp = int(np.sum(r > tol))
        lhp = int(np.sum(r < -tol))
        jw = int(roots.size - rhp - lhp)
        if rhp == 0 and jw == 0:
            desc = "Stable (all poles in LHP)"
        elif rhp == 0 and jw > 0:
            desc = f"Marginally stable: {jw} pole(s) on jω-axis"
        else:
            desc = f"Unstable: {rhp} RHP pole(s), {jw} jω-axis pole(s)"
        return lhp, jw, rhp, desc


    import numpy as np

    def compute_routh(coeffs, tol=1e-12):
        """
        compute_routh with zero-row REPLACEMENT:
        - When a computed row is a full-zero row, replace that row with the
            derivative of the auxiliary polynomial built from the row immediately
            above the zero row (classical replacement rule).
        - Keeps function name and return signature unchanged.
        """
        a = np.trim_zeros(np.asarray(coeffs, dtype=float), 'f')
        if a.size == 0:
            return [], [], 0, 0, 0, "No polynomial provided."

        n = int(a.size - 1)
        cols = int(np.ceil((n + 1) / 2.0))
        rows = n + 1

        # working table (may contain zero rows while building)
        routh = [[0.0 for _ in range(cols)] for _ in range(rows)]

        # populate first two rows (highest-first polynomial coefficients)
        for j in range(cols):
            ei = 2 * j
            oi = 2 * j + 1
            routh[0][j] = a[ei] if ei < a.size else 0.0
            routh[1][j] = a[oi] if oi < a.size else 0.0

        def is_zero_row(row):
            return all(abs(x) < tol for x in row)

        eps = 1e-14
        jw_count = 0
        i = 2

        # build table
        while i < rows:
            Above = routh[i - 1]   # row immediately above (a)
            Top   = routh[i - 2]   # row two above (b)

            # compute the normal recurrence for the current row (candidate row)
            a0 = Above[0]
            if abs(a0) < tol:
                a0 = eps if a0 >= 0 else -eps
                Above[0] = a0
                routh[i - 1][0] = a0

            candidate = [0.0] * cols
            for jcol in range(cols - 1):
                bj1 = Top[jcol + 1] if (jcol + 1) < len(Top) else 0.0
                aj1 = Above[jcol + 1] if (jcol + 1) < len(Above) else 0.0
                numerator = a0 * bj1 - aj1 * Top[0]
                candidate[jcol] = numerator / a0
            if cols > 0:
                candidate[cols - 1] = 0.0

            # If candidate is ALL ZERO -> replace it using auxiliary derivative rule
            if is_zero_row(candidate):
                # Build auxiliary polynomial from the ROW IMMEDIATELY ABOVE the zero row (Above)
                # The row 'Above' corresponds to powers with highest power m = n - (i-1)
                m = n - (i - 1)
                if m < 0:
                    # defensive fallback
                    routh[i] = candidate[:]
                    i += 1
                    continue

                # aux_full: highest-first coefficients of degree m
                aux_full = np.zeros(m + 1, dtype=float)
                for jcol, coef in enumerate(Above):
                    if abs(coef) < tol:
                        continue
                    power = m - 2 * jcol
                    if power < 0:
                        continue
                    idx = m - power  # equals 2*jcol, but keep for clarity
                    if 0 <= idx <= m:
                        aux_full[idx] = coef

                aux_full = np.trim_zeros(aux_full, 'f')
                if aux_full.size == 0:
                    aux_full = np.array([0.0], dtype=float)
                    deg_aux = 0
                else:
                    deg_aux = int(aux_full.size - 1)

                if deg_aux > 0:
                    jw_count += deg_aux

                # derivative of auxiliary (highest-first)
                if deg_aux <= 0:
                    deriv = np.array([0.0], dtype=float)
                else:
                    deriv = np.array([aux_full[k] * (deg_aux - k) for k in range(deg_aux)], dtype=float)

                # map derivative into the columns of the Routh row: deriv[0], deriv[2], deriv[4], ...
                newrow = [0.0] * cols
                for c in range(cols):
                    idx = 2 * c
                    if idx < deriv.size:
                        newrow[c] = deriv[idx]
                    else:
                        newrow[c] = 0.0

                routh[i] = newrow[:]
            else:
                # accept candidate row (no zero replacement needed)
                routh[i] = candidate[:]

            i += 1

        # finished building table; filter zero rows only in the returned view
        routh_out = [row[:] for row in routh if not is_zero_row(row)]

        # build display-version (normalize each row so first nonzero element is positive)
        routh_disp = []
        for row in routh_out:
            first = None
            for v in row:
                if abs(v) >= tol:
                    first = v
                    break
            if first is None:
                disp_row = [0.0 if abs(x) < tol else x for x in row]
            else:
                sgn = 1.0 if first > 0 else -1.0
                disp_row = [(x * sgn) if abs(x) >= tol else 0.0 for x in row]
                disp_row = [int(x) if isinstance(x, float) and abs(x - round(x)) < 1e-9 else x for x in disp_row]
            routh_disp.append(disp_row)

        first_col_disp = [row[0] for row in routh_disp]

        # count sign changes in first column (skip zeros)
        cleaned = [v for v in first_col_disp if abs(v) >= tol]
        sign_changes = 0
        last = None
        for v in cleaned:
            s = 1 if v > 0 else -1
            if last is not None and s != last:
                sign_changes += 1
            last = s

        rhp = sign_changes
        jw_count = min(jw_count, n)
        lhp = n - (rhp + jw_count)

        if rhp == 0 and jw_count == 0:
            stability = "Stable (no RHP or jω poles)"
        elif rhp == 0 and jw_count > 0:
            stability = f"Marginally stable ({jw_count} jω poles)"
        else:
            stability = f"Unstable ({rhp} RHP poles, {jw_count} jω poles)"

        return routh_out, first_col_disp, rhp, jw_count, lhp, stability

#######################################################################################

    # helper: add polynomials (highest-first), return numpy array
    def poly_add(a, b):
        pa = np.asarray(a, dtype=float)
        pb = np.asarray(b, dtype=float)
        # pad the smaller one on the left (highest-first representation)
        if pa.size < pb.size:
            pa = np.pad(pa, (pb.size - pa.size, 0), mode='constant', constant_values=0.0)
        elif pb.size < pa.size:
            pb = np.pad(pb, (pa.size - pb.size, 0), mode='constant', constant_values=0.0)
        return pa + pb

    try:
        # read form values or defaults
        if request.method == 'POST':
            num = parse_poly(request.form.get("num", defaults["num"]))
            den = parse_poly(request.form.get("den", defaults["den"]))
            K_min = float(request.form.get("K_min", defaults["K_min"]))
            K_max = float(request.form.get("K_max", defaults["K_max"]))
            K_steps = int(request.form.get("K_steps", defaults["K_steps"]))
            K_view = float(request.form.get("K_view", defaults["K_view"]))
        else:
            num = parse_poly(defaults["num"])
            den = parse_poly(defaults["den"])
            K_min, K_max, K_steps, K_view = defaults["K_min"], defaults["K_max"], defaults["K_steps"], defaults["K_view"]

        if K_steps < 2:
            raise ValueError("K_steps must be >= 2")

        Ks = np.linspace(K_min, K_max, K_steps)

        # pad numerator to denominator length (numpy arrays)
        if num.size < den.size:
            num = np.pad(num, (den.size - num.size, 0), mode='constant', constant_values=0.0)

        max_order = max(den.size, num.size)
        # store complex nan for empty entries
        all_roots = np.full((K_steps, max_order), np.nan + 1j * np.nan, dtype=complex)

        for i, K in enumerate(Ks):
            denom_cl = np.trim_zeros(poly_add(den, K * num), 'f')
            roots = safe_roots(denom_cl)
            if roots.size:
                all_roots[i, :len(roots)] = roots

        trajs = all_roots.T

        # open-loop poles/zeros and numeric classification
        poles_ol = safe_roots(den) if den.size else np.array([], dtype=complex)
        zeros_ol = safe_roots(num) if num.size else np.array([], dtype=complex)
        lhp_ol_n, jw_ol_n, rhp_ol_n, stability_text_ol_n = classify_poles(poles_ol)

        # Plot root locus
        fig, ax = plt.subplots(figsize=(7, 6))
        for traj in trajs:
            if np.any(np.isfinite(traj)):
                ax.plot(np.real(traj), np.imag(traj), linewidth=1)

        if poles_ol.size > 0:
            ax.scatter(np.real(poles_ol), np.imag(poles_ol), marker='x', s=80, label='Open-loop poles')
        if zeros_ol.size > 0:
            ax.scatter(np.real(zeros_ol), np.imag(zeros_ol),
                       facecolors='none', edgecolors='g', linewidths=1.2, marker='o', s=80, label='Open-loop zeros')

        ax.axhline(0, color='k', linewidth=0.6)
        ax.axvline(0, color='k', linewidth=0.6)
        ax.set_title("Root Locus")
        ax.set_xlabel("Real")
        ax.set_ylabel("Imaginary")
        ax.set_aspect('equal', adjustable='box')
        ax.grid(True, linestyle=':', linewidth=0.6)
        ax.legend()
        fig.tight_layout()
        plot_html += mpld3.fig_to_html(fig)
        plt.close(fig)
        plot_html += "<br><br>"

        # Characteristic polynomial and closed-loop step at K_view
        denom_view = np.trim_zeros(poly_add(den, K_view * num), 'f')
        poles_cl = safe_roots(denom_view)
        lhp_cl, jw_cl, rhp_cl, stability_text_cl = classify_poles(poles_cl)

        try:
            # Attempt to create TransferFunction for diagnostics/plot legend (scipy.signal expects highest-first arrays)
            sys_cl = signal.TransferFunction((K_view * num).tolist(), denom_view.tolist())

            fig2, ax2 = plt.subplots(figsize=(7, 6))
            for traj in trajs:
                if np.any(np.isfinite(traj)):
                    ax2.plot(np.real(traj), np.imag(traj), alpha=0.3)

            if poles_cl.size:
                ax2.scatter(np.real(poles_cl), np.imag(poles_cl),
                            marker='s', s=80, color='red', label=f'CL Poles (K={K_view})')
            if poles_ol.size > 0:
                ax2.scatter(np.real(poles_ol), np.imag(poles_ol), marker='x', s=60)
            if zeros_ol.size > 0:
                ax2.scatter(np.real(zeros_ol), np.imag(zeros_ol),
                            facecolors='none', edgecolors='g', linewidths=1.2, marker='o', s=60)

            ax2.axhline(0, color='k', linewidth=0.6)
            ax2.axvline(0, color='k', linewidth=0.6)
            ax2.set_title(f"Root Locus + Closed-loop poles at K={K_view}")
            ax2.set_aspect('equal', adjustable='box')
            ax2.grid(True, linestyle=':', linewidth=0.6)
            ax2.legend()
            fig2.tight_layout()
            plot_html += mpld3.fig_to_html(fig2)
            plt.close(fig2)
            plot_html += "<br><br>"

        except Exception as ee:
            plot_html += f"<div style='color:red'>Closed-loop error: {ee}</div>"

        # ----------------- Routh array (OPEN-LOOP denominator) -----------------
        try:
            routh_table, first_col, rhp_ol_h, jw_ol_h, lhp_ol_h, stability_text_ol_h = compute_routh(den.tolist())

            rows = len(routh_table)
            cols = len(routh_table[0]) if rows > 0 else 0
            html = "<h3>Routh–Hurwitz Array (from OPEN-LOOP denominator coefficients)</h3>"
            html += f"<p><strong>Polynomial (highest-first):</strong> {' '.join(str(int(x)) if float(x).is_integer() else str(x) for x in den)}</p>"
            html += "<table style='border-collapse:collapse;border:1px solid #ddd;'>"
            html += "<thead><tr><th style='border:1px solid #ddd;padding:6px;background:#f7f7f7'>s<sup>n</sup></th>"
            for c in range(cols):
                html += f"<th style='border:1px solid #ddd;padding:6px;background:#f7f7f7'>c<sub>{c+1}</sub></th>"
            html += "</tr></thead><tbody>"

            degree = len(den) - 1
            for r in range(rows):
                html += "<tr>"
                html += f"<td style='border:1px solid #ddd;padding:6px;background:#f3f9ff'>s<sup>{degree - r}</sup></td>"
                for c in range(cols):
                    val = routh_table[r][c]
                    cell = "0" if abs(val) < 1e-12 else f"{val:.6g}"
                    html += f"<td style='border:1px solid #ddd;padding:6px;text-align:center'>{cell}</td>"
                html += "</tr>"
            html += "</tbody></table>"

            html += f"<p style='margin-top:10px;'><strong>First column:</strong> {' , '.join(f'{x:.6g}' for x in first_col)}</p>"
            html += "<h4>Stability summary (OPEN-LOOP polynomial)</h4>"
            # USE NUMERIC COUNTS (from open-loop roots)
            html += f"<p>LHP poles: <strong>{lhp_ol_n}</strong></p>"
            html += f"<p>jω-axis poles: <strong>{jw_ol_n}</strong></p>"
            html += f"<p>RHP poles: <strong style='color:#b00020'>{rhp_ol_n}</strong></p>"
            html += f"<p style='color:#b00020'><strong>Overall:</strong> {stability_text_ol_n}</p>"

            routh_html = html

        except Exception as re_ol:
            routh_html = f"<div style='color:#b00020'>Routh computation failed (open-loop): {re_ol}</div>"

        # ----------------- Routh for CHARACTERISTIC polynomial at K_view -----------------
        try:
            denom_view_list = denom_view.tolist() if isinstance(denom_view, np.ndarray) else list(denom_view)
            routh_char, first_col_char, rhp_char_h, jw_char_h, lhp_char_h, stability_char_h = compute_routh(denom_view_list)

            rows = len(routh_char)
            cols = len(routh_char[0]) if rows > 0 else 0
            html2 = f"<h3>Routh–Hurwitz (Characteristic polynomial Δ(s) at K = {K_view:+g})</h3>"
            html2 += "<p><strong>Δ(s) coefficients (highest-first):</strong> "
            html2 += " ".join(str(int(x)) if float(x).is_integer() else str(x) for x in denom_view_list)
            html2 += "</p>"

            html2 += "<table style='border-collapse:collapse;border:1px solid #ddd;'>"
            html2 += "<thead><tr><th style='border:1px solid #ddd;padding:6px;background:#f7f7f7'>s<sup>n</sup></th>"
            for c in range(cols):
                html2 += f"<th style='border:1px solid #ddd;padding:6px;background:#f7f7f7'>c<sub>{c+1}</sub></th>"
            html2 += "</tr></thead><tbody>"

            degree_char = len(denom_view_list) - 1
            for r in range(rows):
                html2 += "<tr>"
                html2 += f"<td style='border:1px solid #ddd;padding:6px;background:#f3f9ff'>s<sup>{degree_char - r}</sup></td>"
                for c in range(cols):
                    val = routh_char[r][c]
                    cell = "0" if abs(val) < 1e-12 else f"{val:.6g}"
                    html2 += f"<td style='border:1px solid #ddd;padding:6px;text-align:center'>{cell}</td>"
                html2 += "</tr>"
            html2 += "</tbody></table>"

            html2 += f"<p style='margin-top:10px;'><strong>First column:</strong> {' , '.join(f'{x:.6g}' for x in first_col_char)}</p>"
            html2 += "<h4>Stability summary (Δ(s) at K_view)</h4>"
            # USE NUMERIC COUNTS (from poles_cl)
            html2 += f"<p>LHP poles: <strong>{lhp_cl}</strong></p>"
            html2 += f"<p>jω-axis poles: <strong>{jw_cl}</strong></p>"
            html2 += f"<p>RHP poles: <strong style='color:#b00020'>{rhp_cl}</strong></p>"
            html2 += f"<p style='color:#b00020'><strong>Overall:</strong> {stability_text_cl}</p>"

            routh_html += "<hr style='margin:16px 0'>" + html2

        except Exception as re2:
            routh_html += f"<div style='color:#b00020'>Routh (Δ(s) at K_view) failed: {re2}</div>"

    except Exception as e:
        error = str(e)

    return render_template("rootlocus.html",
                           plot_html=plot_html,
                           routh_html=routh_html,
                           error=error,
                           form_defaults={
                               "num": request.form.get("num", defaults["num"]),
                               "den": request.form.get("den", defaults["den"]),
                               "K_min": request.form.get("K_min", defaults["K_min"]),
                               "K_max": request.form.get("K_max", defaults["K_max"]),
                               "K_steps": request.form.get("K_steps", defaults["K_steps"]),
                               "K_view": request.form.get("K_view", defaults["K_view"])
                           })


###########################################################################################
@app.route('/pulse_delta_mod_and_demod', methods=['GET', 'POST'])
def delta_mod_and_demod():
    plot_html = ""
    error = None

    # Simple defaults (single-sine message)
    defaults = {
        "fs": 1000.0,       # Sampling frequency (Hz)
        "duration": 0.05,   # Duration (s)
        "fm": 5.0,          # Message frequency (Hz)
        "Am": 1.0,          # Message amplitude
        "delta": 0.1        # Step size (Δ)
    }

    try:
        # Read form values (or use defaults)
        if request.method == 'POST':
            fs = float(request.form.get("fs", defaults["fs"]))
            duration = float(request.form.get("duration", defaults["duration"]))
            fm = float(request.form.get("fm", defaults["fm"]))
            Am = float(request.form.get("Am", defaults["Am"]))
            delta = float(request.form.get("delta", defaults["delta"]))
        else:
            fs = defaults["fs"]
            duration = defaults["duration"]
            fm = defaults["fm"]
            Am = defaults["Am"]
            delta = defaults["delta"]

        # basic validation
        if fs <= 0:
            raise ValueError("Sampling frequency must be positive")
        if duration <= 0:
            raise ValueError("Duration must be positive")
        if delta <= 0:
            raise ValueError("Step size (delta) must be positive")

        # Time vector
        t = np.arange(0, duration, 1.0 / fs)
        if t.size == 0:
            raise ValueError("Time vector is empty; increase duration or decrease fs")

        figs = []

        # Message: single sine wave (simple)
        message = Am * np.sin(2 * np.pi * fm * t)

        # 1) Message plot
        fig1, ax1 = plt.subplots(figsize=(8, 3))
        ax1.plot(t, message, linewidth=1)
        ax1.set_title("Message Signal")
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("Amplitude")
        fig1.tight_layout()
        figs.append(fig1)

        # 2) Delta Modulation (encoder + reconstructor)
        N = len(t)
        y = np.zeros(N)    # reconstructor output (demod/reconstructed)
        eq = np.zeros(N)   # step decisions (+Δ or -Δ)

        # initial conditions: start reconstructor at first sample or 0 (choice)
        y[0] = 0.0
        eq[0] = 0.0

        for i in range(1, N):
            # decide step based on previous reconstructor value y[i-1]
            if message[i] > y[i-1]:
                eq[i] = delta
            else:
                eq[i] = -delta
            y[i] = y[i-1] + eq[i]

        # 2a) Show reconstruction levels vs message
        fig2, ax2 = plt.subplots(figsize=(8, 3))
        ax2.plot(t, message, label="Message", linewidth=1)
        ax2.step(t, y, label="Reconstructed (delta steps)", where='post')
        ax2.set_title("Delta Modulated (reconstructor levels)")
        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("Amplitude")
        ax2.legend()
        fig2.tight_layout()
        figs.append(fig2)

        # 3) Demodulated signal plot (use y as demodulated)
        demodulated = y.copy()
        fig3, ax3 = plt.subplots(figsize=(8, 3))
        ax3.plot(t, message, label="Original Message", linewidth=1)
        ax3.plot(t, demodulated, label="Demodulated (reconstructed)", linestyle='--')
        ax3.set_title("Delta Demodulation")
        ax3.set_xlabel("Time (s)")
        ax3.set_ylabel("Amplitude")
        ax3.legend()
        fig3.tight_layout()
        figs.append(fig3)

        # Optional: error plot (message - reconstructed)
        err = message - demodulated
        fig4, ax4 = plt.subplots(figsize=(8, 2.5))
        ax4.plot(t, err)
        ax4.set_title("Error (message - reconstructed)")
        ax4.set_xlabel("Time (s)")
        ax4.set_ylabel("Error")
        fig4.tight_layout()
        figs.append(fig4)

        # Convert figures to HTML and close to free memory
        for f in figs:
            plot_html += mpld3.fig_to_html(f) + "<br><br>"
            plt.close(f)

    except Exception as e:
        error = str(e)

    return render_template('pulse_delta_mod_and_demod.html',
                           plot_html=plot_html, error=error)
################################################################################
@app.route('/pulse_width_mod_and_demod', methods=['GET', 'POST'])
def pwm_mod_and_demod():
    plot_html = ""
    error = None

    # Default parameters
    defaults = {
        "fs": 5000,       # Sampling frequency (Hz)
        "f_pwm": 50,      # PWM carrier frequency (Hz)
        "duration": 0.05, # Signal duration (s)
        "seed": 0         # Random seed for reproducibility
    }

    try:
        # Get inputs from form (or defaults)
        if request.method == 'POST':
            fs = int(request.form.get("fs", defaults["fs"]))
            f_pwm = float(request.form.get("f_pwm", defaults["f_pwm"]))
            duration = float(request.form.get("duration", defaults["duration"]))
            seed = int(request.form.get("seed", defaults["seed"]))
        else:
            fs, f_pwm, duration, seed = defaults.values()

        # Time vector
        t = np.linspace(0, duration, int(fs * duration), endpoint=False)

        # Random waveform
        np.random.seed(seed)
        random_wave = np.random.rand(len(t))  
        random_wave = np.convolve(random_wave, np.ones(50)/50, mode='same')  # smooth

        # Sawtooth carrier for PWM
        carrier = (t * f_pwm) % 1  

        # PWM output
        pwm_output = (random_wave > carrier).astype(float)

        figs = []

        # Plot 1: Random waveform + Carrier
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.plot(t, random_wave, label='Random Waveform')
        ax.plot(t, carrier, label='Carrier (Sawtooth)', alpha=0.7)
        ax.set_title("Random Waveform vs Carrier")
        ax.legend()
        ax.grid(True)
        figs.append(fig)

        # Plot 2: PWM output
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.plot(t, pwm_output, drawstyle='steps-pre', label='PWM Output')
        ax.set_title("PWM Signal")
        ax.legend()
        ax.grid(True)
        figs.append(fig)

        # Plot 3: Overlay
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.plot(t, random_wave, label='Random Waveform')
        ax.plot(t, pwm_output, drawstyle='steps-pre', label='PWM Output')
        ax.set_title("Overlay of Input and PWM")
        ax.legend()
        ax.grid(True)
        figs.append(fig)

        # Convert plots to HTML
        for f in figs:
            plot_html += mpld3.fig_to_html(f) + "<br><br>"

    except Exception as e:
        error = str(e)

    return render_template("pulse_width_mod_and_demod.html", plot_html=plot_html, error=error)
##################################################################################################
class Node:
    def __init__(self, char, freq):
        self.char = char
        self.freq = freq
        self.left = None
        self.right = None

def huffman_encode(message):
    freq = {}
    for ch in message:
        freq[ch] = freq.get(ch, 0) + 1
    nodes = [Node(ch, f) for ch, f in freq.items()]
    while len(nodes) > 1:
        nodes = sorted(nodes, key=lambda x: x.freq)
        left, right = nodes[0], nodes[1]
        newNode = Node(None, left.freq + right.freq)
        newNode.left, newNode.right = left, right
        nodes = [newNode] + nodes[2:]
    root = nodes[0]

    codes = {}
    def generate_codes(node, code=""):
        if node is None: return
        if node.char is not None:
            codes[node.char] = code
        generate_codes(node.left, code+"0")
        generate_codes(node.right, code+"1")
    generate_codes(root)

    encoded = "".join(codes[ch] for ch in message)
    return encoded, codes, root

# Build Huffman Tree Graph for visualization
def build_tree_graph(root):
    G = nx.DiGraph()
    def add_edges(node):
        if node:
            label = f"{node.char}:{node.freq}" if node.char else f"*:{node.freq}"
            if node.left:
                l = f"{node.left.char}:{node.left.freq}" if node.left.char else f"*:{node.left.freq}"
                G.add_edge(label, l, bit="0")
                add_edges(node.left)
            if node.right:
                r = f"{node.right.char}:{node.right.freq}" if node.right.char else f"*:{node.right.freq}"
                G.add_edge(label, r, bit="1")
                add_edges(node.right)
    add_edges(root)
    return G

def huffman_decode(encoded, codes):
    reverse = {v: k for k, v in codes.items()}
    decoded, buff = "", ""
    for bit in encoded:
        buff += bit
        if buff in reverse:
            decoded += reverse[buff]
            buff = ""
    return decoded

# --- RLE ---
def rle_encode(data):
    encoding = ""
    i = 0
    while i < len(data):
        count = 1
        while i + 1 < len(data) and data[i] == data[i+1]:
            i += 1
            count += 1
        encoding += str(count) + data[i]
        i += 1
    return encoding

def rle_decode(data):
    decoded = ""
    count = ""
    for char in data:
        if char.isdigit():
            count += char
        else:
            decoded += char * int(count)
            count = ""
    return decoded

# --- LZW ---
def lzw_compress(uncompressed):
    dict_size = 256
    dictionary = {chr(i): i for i in range(dict_size)}
    w, result = "", []
    for c in uncompressed:
        wc = w + c
        if wc in dictionary:
            w = wc
        else:
            result.append(dictionary[w])
            dictionary[wc] = dict_size
            dict_size += 1
            w = c
    if w:
        result.append(dictionary[w])
    return result

def lzw_decompress(compressed):
    dict_size = 256
    dictionary = {i: chr(i) for i in range(dict_size)}
    result = [dictionary[compressed[0]]]
    w = result[0]
    for k in compressed[1:]:
        if k in dictionary:
            entry = dictionary[k]
        elif k == dict_size:
            entry = w + w[0]
        result.append(entry)
        dictionary[dict_size] = w + entry[0]
        dict_size += 1
        w = entry
    return "".join(result)

def add_noise(data, probability=0.1):
    noisy = ""
    for bit in data:
        if random.random() < probability:
            noisy += "1" if bit == "0" else "0"
        else:
            noisy += bit
    return noisy

@app.route("/channel_coding", methods=["GET", "POST"])
def channel_coding():
    result = None
    error = None
    tree_html = ""

    if request.method == "POST":
        algo = request.form.get("algorithm")
        message = request.form.get("message", "")
        apply_noise = request.form.get("apply_noise")

        if not message:
            error = "Message cannot be empty!"
        else:
            try:
                if algo == "huffman":
                    encoded, codes, root = huffman_encode(message)

                    if apply_noise:
                        noisy = add_noise(encoded)
                        decoded = huffman_decode(noisy, codes)
                        result = f"Huffman Coding\nMessage: {message}\nCodes: {codes}\nEncoded: {encoded}\nNoisy: {noisy}\nDecoded: {decoded}"
                    else:
                        decoded = huffman_decode(encoded, codes)
                        result = f"Huffman Coding\nMessage: {message}\nCodes: {codes}\nEncoded: {encoded}\nDecoded: {decoded}"

                    G = build_tree_graph(root)
                    pos = nx.spring_layout(G)

                    fig, ax = plt.subplots(figsize=(8, 4))
                    nx.draw(G, pos, with_labels=True, font_size=9, ax=ax)
                    edge_labels = nx.get_edge_attributes(G, "bit")
                    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

                    tree_html = mpld3.fig_to_html(fig)

                elif algo == "rle":
                    encoded = rle_encode(message)
                    decoded = rle_decode(encoded)
                    result = f"RLE\nMessage: {message}\nEncoded: {encoded}\nDecoded: {decoded}"

                elif algo == "lzw":
                    encoded = lzw_compress(message)
                    decoded = lzw_decompress(encoded)
                    result = f"LZW\nMessage: {message}\nEncoded: {encoded}\nDecoded: {decoded}"

                else:
                    error = "Unknown algorithm!"

            except Exception as e:
                error = str(e)

    return render_template("channel_coding.html", result=result, error=error, tree_html=tree_html)
###########################################################################################
@app.route('/complex_QAM', methods=['GET', 'POST'])
def complex_QAM():
    import numpy as np
    import matplotlib.pyplot as plt
    import mpld3
    from matplotlib.ticker import ScalarFormatter

    plot_html = ""
    error = None

    # ----------- DEFAULTS (scientific notation) -----------
    defaults = {
        "fs": 4.096e9,
        "fb1": 64e6,
        "fb2": 64e6,
        "A1": 2.0,
        "A2": 1.0,
        "N_fft": 2048,
        "fc": 900e6
    }

    try:
        # ----------- READ USER PARAMS -----------
        if request.method == 'POST':
            fs   = float(request.form.get("fs", defaults["fs"]))
            fb1  = float(request.form.get("fb1", defaults["fb1"]))
            fb2  = float(request.form.get("fb2", defaults["fb2"]))
            A1   = float(request.form.get("A1", defaults["A1"]))
            A2   = float(request.form.get("A2", defaults["A2"]))
            N_fft = int(request.form.get("N_fft", defaults["N_fft"]))
            fc    = float(request.form.get("fc", defaults["fc"]))
        else:
            fs, fb1, fb2, A1, A2, N_fft, fc = defaults.values()

        # ----------- TIME AXIS -----------
        t = np.arange(N_fft) / fs

        # ----------- BASEBAND COMPLEX SIGNAL -----------
        g_complex = A1*np.cos(2*np.pi*fb1*t) + 1j*A2*np.cos(2*np.pi*fb2*t)

        g_fft = np.fft.fft(g_complex, N_fft)
        freqs = np.fft.fftfreq(N_fft, 1/fs)

        # ----------- UP-CONVERSION -----------
        c_complex = np.exp(1j*2*np.pi*fc*t)
        v = g_complex * c_complex         # complex passband
        y = v.real                        # actual transmitted real signal
        y_fft = np.fft.fft(y, N_fft)

        # ----------- DEMODULATION -----------
        demod = np.exp(-1j*2*np.pi*fc*t)
        x = y * demod
        x *= 2.0  # scaling
        x_fft = np.fft.fft(x, N_fft)

        # ----------- LOWPASS FILTER -----------
        cutoff = max(fb1, fb2) * 1.5
        norm_cutoff = cutoff / (fs / 2)

        b = 0.08
        N = int(np.ceil(4/b))
        if N % 2 == 0:
            N += 1

        n = np.arange(N)
        M = (N-1)/2
        h_ideal = norm_cutoff * np.sinc(norm_cutoff * (n - M))
        h = h_ideal * np.blackman(N)
        h /= np.sum(h)

        z = np.convolve(x, h, mode="same")
        z_fft = np.fft.fft(z, N_fft)

        # ----------- PLOT HELPER -----------
        figs = []

        def add_fig(fig):
            figs.append(fig)

        def make_plot(xd, yd, title, xlabel, ylabel):
            fig, ax = plt.subplots(figsize=(8, 3))
            ax.plot(xd, yd)
            ax.set_title(title)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)

            # Enforce scientific notation for time axis
            if xlabel.lower().startswith("time"):
            # Force tick labels into scientific notation manually
                xticks = ax.get_xticks()
                ax.set_xticklabels([f"{x:.2e}" for x in xticks])

            fig.tight_layout()
            add_fig(fig)

        # ----------- GENERATE PLOTS -----------

        # Baseband magnitude
        make_plot(t[:200], np.abs(g_complex[:200]),
                  "Time Domain |g(t)| Baseband Complex", "Time (s)", "Magnitude")

        # FFT one-sided
        f_pos = freqs[:N_fft//2] / 1e6
        make_plot(f_pos, np.abs(g_fft[:N_fft//2]),
                  "One-Sided FFT of Baseband Complex", "Frequency (MHz)", "Magnitude")

        # FFT shift
        f_shift = np.fft.fftshift(freqs)/1e6
        make_plot(f_shift, np.abs(np.fft.fftshift(g_fft)),
                  "Two-Sided FFT of Baseband Complex", "Frequency (MHz)", "Magnitude")

        # Carrier
        make_plot(t[:200], np.abs(c_complex[:200]),
                  "Time Domain |Carrier| Magnitude", "Time (s)", "Magnitude")

        # Modulated (complex magnitude)
        v_fft = np.fft.fft(v, N_fft)
        make_plot(t[:200], np.abs(v[:200]),
                  "Complex Modulated |v(t)|", "Time (s)", "Magnitude")
        make_plot(f_shift, np.abs(np.fft.fftshift(v_fft)),
                  "FFT of Complex Modulated Signal", "Frequency (MHz)", "Magnitude")

        # Real transmitted waveform
        make_plot(t[:200], y[:200],
                  "Real Passband Signal y(t)", "Time (s)", "Amplitude")

        # Demodulated magnitude
        make_plot(t[:200], np.abs(x[:200]),
                  "Demodulated |x(t)|", "Time (s)", "Magnitude")

        # LPF output
        make_plot(t[:200], np.abs(z[:200]),
                  "Lowpass Recovered |z(t)|", "Time (s)", "Magnitude")
        make_plot(f_pos, np.abs(z_fft[:N_fft//2]),
                  "FFT of LPF Output", "Frequency (MHz)", "Magnitude")

        # ----------- RETURN HTML -----------
        for fig in figs:
            plot_html += mpld3.fig_to_html(fig) + "<br><br>"
            plt.close(fig)

    except Exception as e:
        error = str(e)

    return render_template("complex_QAM.html",
                           plot_html=plot_html,
                           error=error,
                           fs=fs, fb1=fb1, fb2=fb2,
                           A1=A1, A2=A2, N_fft=N_fft, fc=fc)

################################################################################################
@app.route("/plots", methods=["GET", "POST"])
def plots():
    import numpy as np, mpld3
    import matplotlib.pyplot as plt
    from scipy import signal

    plots = []

    # ---- Get UI values ----
    N = int(request.form.get("N", 1000))
    sig_type = request.form.get("sig_type", "impulse")
    amp = float(request.form.get("amp", 1))
    shift = float(request.form.get("shift", 0))
    time_scale = float(request.form.get("time_scale", 1))
    step_shift = float(request.form.get("step_shift", 0))
    reverse = request.form.get("reverse") == "on"

    # ---- Generate time ----
    t = np.linspace(-5, 5, N)

    # ---- Original Signal ----
    if sig_type == "impulse":
        x = signal.unit_impulse(N)
    elif sig_type == "step":
        x = np.heaviside(t, 1)
    elif sig_type == "sin":
        x = np.sin(2*np.pi*t)
    elif sig_type == "cos":
        x = np.cos(2*np.pi*t)
    elif sig_type == "square":
        x = signal.square(2*np.pi*t)
    elif sig_type == "sawtooth":
        x = signal.sawtooth(2*np.pi*t)
    elif sig_type == "exp":
        x = np.exp(-0.5*np.abs(t))
    else:
        x = np.zeros_like(t)

    # ---- Apply Signal Operations ----
    t_mod = (t - shift) * time_scale  # shift THEN scale

    if reverse:
        t_mod = -t_mod

    x_mod = amp * np.interp(t_mod, t, x, left=0, right=0)  # interpolation keeps signal shape

    # Add shifted unit step separately: u(t - k)
    step_u = np.heaviside(t - step_shift, 1)

    # ---- Plot helper ----
    def add_plot(y, label):
        fig, ax = plt.subplots()
        ax.plot(t, y)
        ax.set_title(label)
        ax.grid(True)
        plots.append(mpld3.fig_to_html(fig))
        plt.close(fig)

    # ---- Plots ----
    add_plot(x, "Original Signal")
    add_plot(x_mod, "Transformed Signal")
    add_plot(step_u, f"Unit Step u(t - {step_shift})")

    # FFT of transformed signal
    Y = np.fft.fftshift(np.abs(np.fft.fft(x_mod)))
    freqs = np.fft.fftshift(np.fft.fftfreq(N, t[1]-t[0]))
    fig, ax = plt.subplots()
    ax.plot(freqs, Y)
    ax.set_title("Frequency Magnitude Response |X(f)|")
    ax.grid(True)
    plots.append(mpld3.fig_to_html(fig))
    plt.close(fig)

    return render_template("plots.html", plots=plots)

######################################################################################################
@app.route('/control_sys', methods=['GET', 'POST'])
def control_sys():
    plot_html = ""
    error = None   # keep this for exception messages only

    # --- Defaults (simple DC motor + PID) ---
    defaults = {
        "sim_time": 5.0,
        "points": 1000,
        "setpoint": 50.0,      # desired speed (rad/s)
        "Kp": 0.6,
        "Ki": 1.2,
        "Kd": 0.01,
        "J": 0.01,
        "b": 0.1,
        "Kt": 0.01,
        "Ke": 0.01,
        "R": 1.0,
        "V_max": 12.0,
        "V_min": -12.0
    }

    try:
        # read form or defaults
        if request.method == 'POST':
            sim_time = float(request.form.get("sim_time", defaults["sim_time"]))
            points = int(request.form.get("points", defaults["points"]))
            setpoint = float(request.form.get("setpoint", defaults["setpoint"]))
            Kp = float(request.form.get("Kp", defaults["Kp"]))
            Ki = float(request.form.get("Ki", defaults["Ki"]))
            Kd = float(request.form.get("Kd", defaults["Kd"]))
            J = float(request.form.get("J", defaults["J"]))
            b = float(request.form.get("b", defaults["b"]))
            Kt = float(request.form.get("Kt", defaults["Kt"]))
            Ke = float(request.form.get("Ke", defaults["Ke"]))
            R = float(request.form.get("R", defaults["R"]))
            V_max = float(request.form.get("V_max", defaults["V_max"]))
            V_min = float(request.form.get("V_min", defaults["V_min"]))
        else:
            sim_time = defaults["sim_time"]
            points = defaults["points"]
            setpoint = defaults["setpoint"]
            Kp = defaults["Kp"]; Ki = defaults["Ki"]; Kd = defaults["Kd"]
            J = defaults["J"]; b = defaults["b"]; Kt = defaults["Kt"]
            Ke = defaults["Ke"]; R = defaults["R"]
            V_max = defaults["V_max"]; V_min = defaults["V_min"]

        # sanity checks
        points = max(10, int(points))
        sim_time = max(1e-6, float(sim_time))

        # time base and storage
        t = np.linspace(0, sim_time, points)
        dt = t[1] - t[0]

        omega = np.zeros_like(t)      # motor speed (rad/s)
        torque = np.zeros_like(t)     # torque (N·m)
        control = np.zeros_like(t)    # control signal (applied voltage V)
        err_hist = np.zeros_like(t)   # rename: avoid clashing with `error` used for exceptions

        # PID internal states
        integral = 0.0
        prev_err = 0.0

        # initial condition
        omega[0] = 0.0

        # simulation loop: simple Euler integration
        for k in range(1, len(t)):
            # compute error (setpoint - measured speed)
            err = setpoint - omega[k-1]
            err_hist[k] = err

            # PID discrete
            integral += err * dt
            derivative = (err - prev_err) / dt if dt > 0 else 0.0
            V_unclamped = Kp * err + Ki * integral + Kd * derivative

            # clamp to actuator limits
            V = np.clip(V_unclamped, V_min, V_max)
            # simple anti-windup
            if V != V_unclamped:
                integral -= err * dt  # undo last integration

            control[k] = V
            prev_err = err

            # electrical current with back-EMF
            i = (V - Ke * omega[k-1]) / R
            tau = Kt * i
            torque[k] = tau

            # mechanical dynamics
            domega_dt = (tau - b * omega[k-1]) / J
            omega[k] = omega[k-1] + domega_dt * dt

        # --- Plots ---
        figs = []

        # 1) Speed vs time (setpoint vs measured)
        fig, ax = plt.subplots()
        ax.plot(t, np.ones_like(t) * setpoint, 'b--', label='Setpoint')
        ax.plot(t, omega, 'r-', label='Measured speed')
        ax.set_xlabel('Time [s]'); ax.set_ylabel('Speed (rad/s)')
        ax.set_title('Motor Speed: Setpoint vs Measured')
        ax.legend()
        fig.tight_layout()
        figs.append(fig)

        # 2) Control voltage over time
        fig, ax = plt.subplots()
        ax.plot(t, control, label='Control voltage (V)')
        ax.set_xlabel('Time [s]'); ax.set_ylabel('Voltage [V]')
        ax.set_title('PID Control Output (Voltage)')
        fig.tight_layout()
        figs.append(fig)

        # 3) Error over time
        fig, ax = plt.subplots()
        ax.plot(t, err_hist, label='Speed error (set - measured)')
        ax.set_xlabel('Time [s]'); ax.set_ylabel('Error (rad/s)')
        ax.set_title('PID Error')
        fig.tight_layout()
        figs.append(fig)

        # convert figures to HTML
        plot_html = "".join([mpld3.fig_to_html(f) + "<br><br>" for f in figs])

    except Exception as e:
        # set the template `error` variable to a string message only
        error = str(e)

    return render_template('control_sys.html', plot_html=plot_html, error=error, defaults=defaults)

#################################################################################################
@app.route("/DSP", methods=["GET", "POST"])
def DSP():
    result = None
    if request.method == "POST":
        try:
            # ----------- Get User Inputs -----------
            x = list(map(int, request.form["x_seq"].split()))
            y = list(map(int, request.form["y_seq"].split()))
            expr_input = request.form["system_expr"]

            x = np.array(x)
            y = np.array(y)

            # ----------- Convolution -----------
            lin_conv = np.convolve(x, y, mode="full")

            N = max(len(x), len(y))
            x_circ = np.pad(x, (0, N - len(x)), mode='constant')
            y_circ = np.pad(y, (0, N - len(y)), mode='constant')
            circ_conv = np.fft.ifft(np.fft.fft(x_circ) * np.fft.fft(y_circ)).real.round().astype(int)

            # ----------- System Expression -----------
            x_sym = sp.Symbol('x')
            system_expr = sp.sympify(expr_input)
            system_func = sp.lambdify(x_sym, system_expr, modules="numpy")

            # ----------- Linearity Check -----------
            x1, x2, a1, a2 = 1, 2, 3, 4  # fixed test values
            y1 = system_func(x1)
            y2 = system_func(x2)
            lhs = system_func(a1 * x1 + a2 * x2)
            rhs = a1 * y1 + a2 * y2
            linearity = "LINEAR ✅" if np.isclose(lhs, rhs) else "NOT LINEAR ❌"

            # ----------- Time Invariance Check -----------
            t0 = 2
            test_vals = [0, 1, 2, 3, 4, 5]
            passed = True
            for val in test_vals:
                y_shifted_input = system_func(val - t0)
                y_shifted_output = system_func(val - t0)
                if not np.isclose(y_shifted_input, y_shifted_output):
                    passed = False
                    break
            time_invariance = "TIME INVARIANT ✅" if passed else "TIME VARYING ❌"

            # ----------- BIBO Stability Check -----------
            bounded_input = np.linspace(-5, 5, 200)
            outputs = system_func(bounded_input)
            bibo = "BIBO STABLE ✅" if np.all(np.abs(outputs) < 1e6) else "NOT BIBO STABLE ❌"

            # ----------- Collect Results -----------
            result = {
                "lin_conv": lin_conv.tolist(),
                "circ_conv": circ_conv.tolist(),
                "system_expr": str(system_expr),
                "linearity": linearity,
                "time_invariance": time_invariance,
                "bibo": bibo
            }

        except Exception as e:
            result = {"error": str(e)}

    return render_template("DSP.html", result=result)
#################################################################################################
@app.route("/DFT", methods=["GET", "POST"])
def DFT():
    result = None
    if request.method == "POST":
        try:
            # Get user input
            seq_str = request.form["sequence"]
            # allow commas or whitespace separated numbers
            seq_str = seq_str.replace(",", " ")
            x = np.array(list(map(float, seq_str.split())))
            N = len(x)

            # DFT
            X = np.fft.fft(x, N)

            # IDFT
            x_reconstructed = np.fft.ifft(X, N)

            # Prepare textual results (explicit real/imag formatting)
            dft_list = [
                f"k={k}: {X[k].real:.4f} + j{X[k].imag:.4f}" for k in range(N)
            ]
            idft_list = [
                f"n={n}: {x_reconstructed[n].real:.4f} + j{x_reconstructed[n].imag:.4f}"
                for n in range(N)
            ]

            # --- Create plot: magnitude and phase ---
# --- Create plot: magnitude and phase ---
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 5), constrained_layout=True)

            k = np.arange(N)
            magnitude = np.abs(X)
            phase = np.angle(X)

            # Stem without use_line_collection (older Matplotlib fix)
            ax1.stem(k, magnitude, basefmt=" ")
            ax1.set_title("DFT Magnitude")
            ax1.set_xlabel("k")
            ax1.set_ylabel("|X(k)|")
            ax1.grid(True)

            ax2.stem(k, phase, basefmt=" ")
            ax2.set_title("DFT Phase")
            ax2.set_xlabel("k")
            ax2.set_ylabel("Angle (radians)")
            ax2.grid(True)


            # Save plot to PNG in-memory
            buf = io.BytesIO()
            plt.savefig(buf, format="png", dpi=120)
            plt.close(fig)
            buf.seek(0)
            img_base64 = base64.b64encode(buf.read()).decode("utf-8")
            buf.close()

            # Prepare result dict for template
            result = {
                "sequence": x.tolist(),
                "dft": dft_list,
                "idft": idft_list,
                # data-uri (only the base64 payload, embed as data:image/png;base64,{{result.plot}})
                "plot": img_base64
            }

        except Exception as e:
            result = {"error": str(e)}

    return render_template("DFT.html", result=result)
####################################################################################################
@app.route("/filters", methods=["GET", "POST"])
def filters():
    result = ""
    if request.method == "POST":
        try:
            contin = int(request.form.get("contin"))  # 0=continuous, 1=discrete

            if contin == 1:  # discrete filters
                coeff_mode = int(request.form.get("coeff_mode"))  # 1=coeff, 0=ZPK
                if coeff_mode == 1:
                    num = request.form.get("num", "")
                    den = request.form.get("den", "")
                    b = tuple(map(float, num.strip().split()))
                    a = tuple(map(float, den.strip().split()))
                    fil = LTIFilter(b, a)
                else:
                    zeros = tuple(map(lambda z: complex(z), request.form.get("zeros", "").split()))
                    poles = tuple(map(lambda p: complex(p), request.form.get("poles", "").split()))
                    gain = float(request.form.get("gain", "1"))
                    fil = LTIFilter.from_ZPK(zeros, poles, gain)

                result += f"<pre>{fil.transfer_function()}</pre>"
                result += f"<pre>{fil.frequency_response()}</pre>"
                result += f"<pre>{fil.angular_frequency_response()}</pre>"
                result += f"<pre>{fil.group_delay()}</pre>"
                result += f"<pre>{fil.impulse_response()}</pre>"
                result += f"<pre>{fil.step_response()}</pre>"

            else:  # continuous filters
                filt_type = int(request.form.get("filt_type"))  # 0=Bessel, 1=Butterworth
                omega = float(request.form.get("omega", "1"))

                if filt_type == 1:
                    B = Butterworth(N=2, Wn=omega, btype="lowpass")
                else:
                    B = Bessel(N=2, Wn=omega, btype="lowpass")

                result += f"<pre>{B.transfer_function()}</pre>"
                result += f"<pre>{B.frequency_response()}</pre>"
                result += f"<pre>{B.group_delay()}</pre>"

        except Exception as e:
            result = f"<p style='color:red'>Error: {e}</p>"

    return render_template("filters.html", result=result)

###############################################################################################
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Default image path
DEFAULT_IMAGE = os.path.join(UPLOAD_FOLDER, "1.webp")

def load_image():
    img = cv2.imread(DEFAULT_IMAGE)
    if img is None:
        raise FileNotFoundError("Upload '1.webp' into uploads/ folder.")
    img = cv2.copyMakeBorder(img, 50, 50, 50, 50, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

@app.route("/image_processing", methods=["GET", "POST"])
def image_processing():
    result_img = None
    operation = None

    if request.method == "POST":
        operation = request.form.get("operation")
        image_rgb = load_image()
        height, width = image_rgb.shape[:2]

        if operation == "scale":
            sx_zoom = float(request.form.get("sx_zoom", 1.5))
            sy_zoom = float(request.form.get("sy_zoom", 1.5))
            zoomed = cv2.resize(image_rgb, (int(width * sx_zoom), int(height * sy_zoom)), interpolation=cv2.INTER_CUBIC)
            result_img = zoomed

        elif operation == "rotate":
            angle = float(request.form.get("angle", 45))
            scale = float(request.form.get("scale", 1.0))
            center = (width // 2, height // 2)
            matrix = cv2.getRotationMatrix2D(center, angle, scale)
            rotated = cv2.warpAffine(image_rgb, matrix, (width, height))
            result_img = rotated

        elif operation == "translate":
            tx = int(request.form.get("tx", 50))
            ty = int(request.form.get("ty", 30))
            matrix = np.array([[1, 0, tx], [0, 1, ty]], dtype=np.float32)
            translated = cv2.warpAffine(image_rgb, matrix, (width, height))
            result_img = translated

        elif operation == "shear":
            shearX = float(request.form.get("shearX", 0.2))
            shearY = float(request.form.get("shearY", 0.2))
            matrix = np.array([[1, shearX, 0], [0, 1, shearY]], dtype=np.float32)
            sheared = cv2.warpAffine(image_rgb, matrix, (width, height))
            result_img = sheared

        elif operation == "normalize":
            b, g, r = cv2.split(image_rgb)
            b = cv2.normalize(b.astype('float'), None, 0, 1, cv2.NORM_MINMAX)
            g = cv2.normalize(g.astype('float'), None, 0, 1, cv2.NORM_MINMAX)
            r = cv2.normalize(r.astype('float'), None, 0, 1, cv2.NORM_MINMAX)
            normalized = cv2.merge((b, g, r))
            result_img = (normalized * 255).astype(np.uint8)

        elif operation == "edges":
            t1 = int(request.form.get("t1", 100))
            t2 = int(request.form.get("t2", 200))
            edges = cv2.Canny(image_rgb, t1, t2)
            result_img = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)

        elif operation == "blur":
            blurred = cv2.GaussianBlur(image_rgb, (3, 3), 0)
            result_img = blurred

        elif operation == "morphology":
            gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
            kernel = np.ones((3, 3), np.uint8)
            dilated = cv2.dilate(gray, kernel, iterations=2)
            result_img = cv2.cvtColor(dilated, cv2.COLOR_GRAY2RGB)

        if result_img is not None:
            # ✅ Return processed image directly (no matplotlib, no white space)
            _, buffer = cv2.imencode(".png", cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR))
            return send_file(BytesIO(buffer), mimetype="image/png")

    return render_template("image_processing.html")
    


# Simple phasor function using your original logic
def phasor_value(magnitude=0, phase=0):
    """Return complex phasor from magnitude and phase (degrees)"""
    return magnitude * np.exp(1j * math.radians(phase))

# Dummy voltage function for compatibility
def voltage(P):
    """Return object with magnitude, phase, rms for display"""
    class V:
        def __init__(self, P):
            self.magnitude = abs(P)
            self.phase = np.angle(P)
            self.rms = abs(P)/np.sqrt(2)
    return V(P)

# Dummy noise voltage generator
class Noise:
    _nid = 0
    def __init__(self, val):
        self.val = val
        self.nid = Noise._nid
        Noise._nid += 1
    def __add__(self, other):
        return Noise(self.val + other.val)
    def __str__(self):
        return str(self.val)

def noisevoltage(val):
    return Noise(val)

# ===========================
# /index1 route
# ===========================
@app.route("/index1", methods=["GET", "POST"])
def index1():
    results = []
    plot_html = ""
    error = None

    if request.method == "POST":
        try:
            mode = (request.form.get("mode") or "phasor").lower()

            # ---- PHASOR MODE ----
            if mode == "phasor":
                mag = float(request.form.get("mag", 0))
                angle = float(request.form.get("angle", 0))
                freq = float(request.form.get("freq", 50))

                P = phasor_value(magnitude=mag, phase=angle)
                V = voltage(P)

                results.append(f"Mode: PHASOR")
                results.append(f"Input: {mag} ∠ {angle}° @ {freq} Hz")
                results.append(f"Magnitude: {V.magnitude}")
                results.append(f"Phase (rad): {V.phase}")
                results.append(f"RMS: {V.rms}")

                # ---- PLOT ----
                t = np.linspace(0, 0.04, 1000)
                y = mag * np.cos(2*np.pi*freq*t + math.radians(angle))
                plt.figure()
                plt.plot(t, y)
                plt.title(f"Phasor: {mag}∠{angle}° at {freq}Hz")
                plt.xlabel("Time (s)")
                plt.ylabel("Amplitude")
                plot_html = mpld3.fig_to_html(plt.gcf())
                plt.close()

            # ---- WAVEFORM MODE ----
            elif mode == "waveform":
                A = float(request.form.get("A", 5))
                w = float(request.form.get("omega", 314))
                phi = float(request.form.get("phi", 0))

                P = phasor_value(magnitude=A, phase=math.degrees(phi))
                V = voltage(P)

                results.append(f"Mode: WAVEFORM")
                results.append(f"Amplitude: {A}")
                results.append(f"Phase (rad): {phi}")
                results.append(f"Magnitude: {V.magnitude}")
                results.append(f"RMS: {V.rms}")
                results.append(f"Omega: {w}")

                t = np.linspace(0, 0.04, 1000)
                y = A * np.cos(w * t + phi)
                plt.figure()
                plt.plot(t, y)
                plt.title("Time Domain Waveform")
                plt.xlabel("Time (s)")
                plt.ylabel("Amplitude")
                plot_html = mpld3.fig_to_html(plt.gcf())
                plt.close()

            else:
                raise ValueError("Invalid mode. Use 'phasor' or 'waveform'.")

            # ---- Noise ----
            noise_val = float(request.form.get("noise_val", 2))
            X = noisevoltage(noise_val)
            Y = noisevoltage(4)
            Z = X + Y
            results.append("=== Noise ===")
            results.append(f"X: {X}, id={X.nid}")
            results.append(f"Y: {Y}, id={Y.nid}")
            results.append(f"X + Y = {Z}")

        except Exception as e:
            error = str(e)

    return render_template("index1.html", results=results, error=error, plot_html=plot_html)

#######################################################################################################

@app.route('/qpsk', methods=['GET', 'POST'])
def qpsk():
    plot_htmls = []
    error = None
    summary = None

    if request.method == 'POST':
        try:
            num_symbols = int(request.form.get('num_symbols', 1000))
            noise_power = float(request.form.get('noise_power', 0.01))
            phase_noise_strength = float(request.form.get('phase_noise', 0.1))

            # --- QPSK Symbol Generation ---
            x_int = np.random.randint(0, 4, num_symbols)
            x_degrees = x_int * 360 / 4.0 + 45
            x_radians = x_degrees * np.pi / 180.0
            x_symbols = np.cos(x_radians) + 1j * np.sin(x_radians)

            # --- Plot Setup ---
            fig, axs = plt.subplots(3, 2, figsize=(12, 9), constrained_layout=True)
            axs = axs.flatten()

            def add_boundaries(ax):
                ax.axhline(0, color='gray', linestyle='--')
                ax.axvline(0, color='gray', linestyle='--')
                ax.grid(True)

            # 1. Ideal Constellation
            axs[0].plot(np.real(x_symbols), np.imag(x_symbols), '.')
            add_boundaries(axs[0])
            axs[0].set_title("QPSK Constellation (Ideal)")

            # 2. With AWGN Noise
            n = (np.random.randn(num_symbols) + 1j * np.random.randn(num_symbols)) / np.sqrt(2)
            r_awgn = x_symbols + n * np.sqrt(noise_power)
            axs[1].plot(np.real(r_awgn), np.imag(r_awgn), '.')
            add_boundaries(axs[1])
            axs[1].set_title("QPSK with AWGN Noise")

            # 3. With Phase Noise
            phase_noise = np.random.randn(len(x_symbols)) * phase_noise_strength
            r_phase = x_symbols * np.exp(1j * phase_noise)
            axs[2].plot(np.real(r_phase), np.imag(r_phase), '.')
            add_boundaries(axs[2])
            axs[2].set_title("QPSK with Phase Noise")

            # 4. With AWGN + Phase noise
            total_noise = r_phase + n * np.sqrt(noise_power)
            axs[3].plot(np.real(total_noise), np.imag(total_noise), '.')
            add_boundaries(axs[3])
            axs[3].set_title("QPSK with AWGN + Phase Noise")

            # 5. Time-domain I/Q Components
            t = np.arange(0, 100)
            axs[4].plot(t, np.real(x_symbols[:100]), label='I(t)')
            axs[4].plot(t, np.imag(x_symbols[:100]), label='Q(t)')
            axs[4].grid(True)
            axs[4].legend()
            axs[4].set_title("Time-Domain I/Q Components")

            # 6. BER vs SNR
            const_points = np.array([
                np.cos(theta) + 1j*np.sin(theta)
                for theta in (np.array([45,135,225,315]) * np.pi/180.0)
            ])
            idx2bits = {
                0: np.array([0,0]),
                1: np.array([0,1]),
                2: np.array([1,1]),
                3: np.array([1,0])
            }

            bits_tx = np.vstack([idx2bits[i] for i in x_int]).reshape(-1)

            snr_db = np.arange(-5, 15, 2)
            ber = []

            for snr in snr_db:
                noise_var = 10 ** (-snr / 10.0)
                n = (np.random.randn(num_symbols) + 1j * np.random.randn(num_symbols)) / np.sqrt(2)
                r = x_symbols + n * np.sqrt(noise_var)

                dists = np.abs(r.reshape(-1, 1) - const_points.reshape(1, -1))
                detected_idx = np.argmin(dists, axis=1)

                bits_rx = np.vstack([idx2bits[i] for i in detected_idx]).reshape(-1)
                ber.append(np.mean(bits_tx != bits_rx))

            axs[5].semilogy(snr_db, ber, 'o-', label='Simulated BER')
            axs[5].grid(True, which='both')
            axs[5].set_xlabel("SNR (dB)")
            axs[5].set_ylabel("BER")
            axs[5].legend()
            axs[5].set_title("BER vs SNR")

            # Export static plot
            buf = io.BytesIO()
            fig.savefig(buf, format="png", bbox_inches="tight")
            buf.seek(0)
            img = base64.b64encode(buf.read()).decode('utf-8')
            plt.close(fig)
            plot_htmls.append(f'<img src="data:image/png;base64,{img}">')

        except Exception as e:
            error = str(e)

    return render_template('qpsk.html', plots=plot_htmls, summary=summary, error=error)


########################################################################################
def matrix_to_float_np(mat):
    return np.array(Matrix(mat).applyfunc(lambda x: float(x.evalf()))).astype(np.float64)

def create_plot():
    img = io.BytesIO()
    plt.savefig(img, format='png', dpi=120, bbox_inches='tight')
    img.seek(0)
    return base64.b64encode(img.getvalue()).decode()

@app.route('/state_space', methods=['GET','POST'])
def state_space():
    plots = []
    error = None
    system_info = ""   # ✅ display system
    stability_msg = "" # ✅ stability
    ctrl_msg = ""      # ✅ controllability
    obs_msg = ""       # ✅ observability

    if request.method == 'POST':
        try:
            mode = request.form['mode']

            # ---------- Build system ----------
            if mode == '0':
                A = sympify(request.form['A']); B = sympify(request.form['B'])
                C = sympify(request.form['C']); D = sympify(request.form['D'])

                A_np = matrix_to_float_np(A)
                B_np = matrix_to_float_np(B)
                C_np = matrix_to_float_np(C)
                D_np = matrix_to_float_np(D)

                sys = ct.ss(A_np, B_np, C_np, D_np)

                system_info = f"A = {A_np}<br>B = {B_np}<br>C = {C_np}<br>D = {D_np}"

            elif mode == '1':
                b = list(map(float, request.form['b'].split()))
                a = list(map(float, request.form['a'].split()))
                sys = ct.tf(b, a)
                sys = ct.ss(sys)

                system_info = "Transfer Function:<br>Numerator = {}<br>Denominator = {}".format(b, a)

                        # ---------- Stability ----------
            # ---------- Stability ----------
            # ---------- Stability ----------
            poles = sys.poles()   # ✅ corrected
            if np.all(np.real(poles) < 0):
                stability_msg = "✅ System is **Stable** (all poles in left half plane)"
            else:
                stability_msg = "❌ System is **Unstable** (poles in right half plane or on jω-axis)"

            # ---------- Controllability ----------
            Ctrb = ct.ctrb(sys.A, sys.B)
            if np.linalg.matrix_rank(Ctrb) == sys.A.shape[0]:
                ctrl_msg = "✅ System is **Controllable**"
            else:
                ctrl_msg = "⚠️ System is **Not Fully Controllable**"

            # ---------- Observability ----------
            Obsv = ct.obsv(sys.A, sys.C)
            if np.linalg.matrix_rank(Obsv) == sys.A.shape[0]:
                obs_msg = "✅ System is **Observable**"
            else:
                obs_msg = "⚠️ System is **Not Fully Observable**"

            # ---------- Pole-Zero Plot ----------
            fig2, ax2 = plt.subplots(figsize=(5, 5))

            zeros = sys.zeros()   # ✅ corrected
            for p in poles:
                ax2.plot(np.real(p), np.imag(p), 'x', markersize=10)
            for z in zeros:
                ax2.plot(np.real(z), np.imag(z), 'o', markersize=8)

            ax2.axhline(0); ax2.axvline(0)
            ax2.set_title("Pole-Zero Plot")
            ax2.set_xlabel("Real")
            ax2.set_ylabel("Imag")
            ax2.grid()



            buf2 = io.BytesIO()
            fig2.savefig(buf2, format="png", bbox_inches="tight"); buf2.seek(0)
            plots.append(base64.b64encode(buf2.read()).decode("utf-8"))
            plt.close(fig2)

        except Exception as e:
            error = str(e)

    return render_template(
        'state_space.html', 
        plots=plots, 
        error=error,
        system_info=system_info,
        stability_msg=stability_msg,
        ctrl_msg=ctrl_msg,
        obs_msg=obs_msg
    )

##############################################################################################
from lcapy import nexpr, z
import traceback
#################################################################################################
# at top of file (add if not present)
from flask import request, render_template, flash
# lcapy imports
from lcapy import expr
from lcapy.discretetime import n, z

# route (replace/insert into your app)
@app.route('/z_transform', methods=['GET', 'POST'])
def z_transform():
    """
    Unified page to compute Z-transform or inverse Z-transform using Lcapy.
    Form fields expected:
      - signal_expr : string (the expression to transform / inverse-transform)
      - operation   : 'zt' or 'izt'
      - causal      : optional checkbox (for inverse; 'on' or absent)
    """
    results = []
    error = None
    examples = {
        'zt_examples': [
            'u(n)',          # unit step
            '0.5**n * u(n)', # causal exponential
            "delta(n) + 2*delta(n-2)",
            'a**n * u(n)'
        ],
        'izt_examples': [
            'z/(z-a)',
            '1/(1 - 0.5*z**-1)',
            'z/(z-0.5)',     # simple pole form
            '(z**2)/(z**2 - 0.3*z - 0.1)'
        ]
    }

    if request.method == 'POST':
        try:
            user_input = request.form.get('signal_expr', '').strip()
            if not user_input:
                raise ValueError("Input cannot be empty.")

            operation = request.form.get('operation', 'zt').lower()
            causal = request.form.get('causal') == 'on'

            # Parse with lcapy.expr() — recognizes n and z, delta(), u(), etc.
            E = expr(user_input)

            # Show parsed input
            results.append({
                'description': 'Input (parsed)',
                'expr': str(E),
                'latex': getattr(E, 'latex', lambda: str(E))()
            })

            if operation == 'zt':
                # Forward Z-transform: use E.ZT() if available, otherwise call with z
                try:
                    if hasattr(E, 'ZT'):
                        X = E.ZT()
                    else:
                        X = E(z)   # call form
                except Exception:
                    # fallback to call form again
                    X = E(z)

                # Some expressions have simplify()
                X_simpl = getattr(X, 'simplify', lambda: X)()
                results.append({
                    'description': 'Z-transform (unilateral by default)',
                    'expr': str(X_simpl),
                    'latex': getattr(X_simpl, 'latex', lambda: str(X_simpl))()
                })

            elif operation == 'izt':
                # Inverse Z-transform: prefer IZT() if available or call with n
                try:
                    if hasattr(E, 'IZT'):
                        seq_result = E.IZT()
                    else:
                        # use call form X(n) — causal flag if requested
                        seq_result = E(n, causal=causal) if causal else E(n)
                except Exception:
                    # try alternate fallback
                    if hasattr(E, 'IZT'):
                        seq_result = E.IZT()
                    else:
                        # re-raise original error
                        raise

                results.append({
                    'description': 'Inverse Z-transform (sequence)',
                    'expr': str(seq_result),
                    'latex': getattr(seq_result, 'latex', lambda: str(seq_result))()
                })

            else:
                raise ValueError("Unknown operation. Use 'zt' or 'izt'.")

        except Exception as e:
            # include traceback in development mode to aid debugging
            error = f"{e}\n{traceback.format_exc()}"

    return render_template('z_transform.html',
                           results=results,
                           error=error,
                           examples=examples)

###########################################################################################
# correct imports (put at top of your file)
from flask import request, render_template, flash
from lcapy import expr, s, t

# route (replace your previous laplace_transform route with this)
@app.route('/laplace_transform', methods=['GET', 'POST'])
def laplace_transform():
    """
    Compute Laplace transform (LT) or inverse Laplace transform (ILT) using Lcapy.
    Form fields expected:
      - signal_expr : string (expression to transform / inverse-transform)
      - operation   : 'lt' or 'ilt'
      - causal      : optional checkbox (for ILT; passed as causal=True if checked)
    """
    results = []
    error = None

    examples = {
        'lt_examples': [
            'u(t)',               # Heaviside / unit step
            'exp(-2*t) * u(t)',
            'sin(2*t) * u(t)',
            't * u(t)'
        ],
        'ilt_examples': [
            '1/(s + 2)',
            's/(s**2 + 4)',
            '1/(s*(s+1))',
            '(s+1)/(s**2 + 2*s + 2)'
        ]
    }

    if request.method == 'POST':
        try:
            user_input = request.form.get('signal_expr', '').strip()
            if not user_input:
                raise ValueError("Input cannot be empty.")

            operation = request.form.get('operation', 'lt').lower()
            causal = request.form.get('causal') == 'on'

            # Parse expression string; expr() returns an lcapy expression object
            E = expr(user_input)

            # Show parsed input
            results.append({
                'description': 'Input (parsed)',
                'expr': str(E),
                'latex': getattr(E, 'latex', lambda: str(E))()
            })

            if operation == 'lt':
                # Prefer the LT() method; fallback to call-form E(s)
                try:
                    if hasattr(E, 'LT'):
                        X = E.LT()
                    else:
                        X = E(s)
                except Exception:
                    X = E(s)

                X_simpl = getattr(X, 'simplify', lambda: X)()
                results.append({
                    'description': 'Laplace transform (unilateral by default)',
                    'expr': str(X_simpl),
                    'latex': getattr(X_simpl, 'latex', lambda: str(X_simpl))()
                })

            elif operation == 'ilt':
                # Prefer ILT(); otherwise use call-form E(t, causal=True/False)
                try:
                    if hasattr(E, 'ILT'):
                        time_result = E.ILT(causal=causal) if causal else E.ILT()
                    else:
                        time_result = E(t, causal=causal) if causal else E(t)
                except Exception:
                    # last-resort fallbacks
                    if hasattr(E, 'ILT'):
                        time_result = E.ILT()
                    else:
                        time_result = E(t)

                results.append({
                    'description': 'Inverse Laplace transform (time-domain)',
                    'expr': str(time_result),
                    'latex': getattr(time_result, 'latex', lambda: str(time_result))()
                })

            else:
                raise ValueError("Unknown operation. Use 'lt' or 'ilt'.")

        except Exception as e:
            # show traceback in development mode to help debugging
            error = f"{e}\n{traceback.format_exc()}"

    return render_template('laplace_transform.html',
                           results=results,
                           error=error,
                           examples=examples)

##################################################################################################
# Descriptions for each plot
# create a 's' TransferFunction for users to write expressions like "s/(s+2)"
s = ct.tf([1, 0], [1])

PLOT_DESCRIPTIONS = {
    "1": "Pole-zero plot",
    "2": "Bode plot",
    "3": "Nyquist plot",
    "4": "Nichols plot",
}

@app.route("/transfer", methods=["GET", "POST"])
def transfer_page():
    results = []
    error = None

    if request.method == "POST":
        tf_expr = request.form.get("tf_expr", "").strip()
        choices = request.form.getlist("plots")

        if not tf_expr:
            error = "Please enter a transfer function (example: s/(s+2))."
        else:
            try:
                # --- SAFE eval environment with helpful constructors ---
                safe_locals = {
                    "s": s,
                    "tf": ct.tf,
                    "zpk": ct.zpk,
                    "ss": ct.ss,
                    "np": np,
                }
                # disable builtins for safety
                H = eval(tf_expr, {"__builtins__": None}, safe_locals)

                # Validate H is a control system
                if not isinstance(H, (ct.TransferFunction, ct.StateSpace)):
                    # try wrapping if user gave simple numeric gain
                    try:
                        H = ct.tf(float(H))
                    except Exception:
                        raise ValueError("Parsed expression is not a known control system.")

                # For each requested plot, draw on a fresh figure, capture HTML
                for choice in choices:
                    plt.figure(figsize=(6, 4))

                    if choice == "1":
                        # Pole-zero map (ctrl.pzmap draws on current axes)
                        ct.pzmap(H, plot=True, title="Pole-Zero Map")

                    elif choice == "2":
                        # Bode plot: magnitude (dB) & phase vs omega
                        # ctrl.bode_plot returns (mag, phase, omega) but draws figure
                        mag, phase, omega = ct.bode_plot(H, dB=True, plot=True)
                        # optionally adjust layout
                        plt.tight_layout()

                    elif choice == "3":
                        # Nyquist
                        ct.nyquist_plot(H, omega=np.logspace(-2, 2, 500), plot=True)

                    elif choice == "4":
                        # Nichols
                        ct.nichols_plot(H, omega=np.logspace(-2, 2, 500), grid=True)
                    else:
                        plt.close()  # nothing to do
                        continue

                    # Grab current figure and convert to HTML
                    fig = plt.gcf()
                    plot_html = mpld3.fig_to_html(fig)
                    results.append({
                        "description": PLOT_DESCRIPTIONS.get(choice, ""),
                        "plot_html": plot_html
                    })

                    plt.close(fig)  # cleanup

            except Exception as e:
                error = f"Error: {str(e)}"

    return render_template("transfer.html", results=results, error=error)

##################################################################################
def compute_roc(poles, signal_type):
    """
    Robust ROC computation for causal / anti-causal / two-sided signals.
    Returns (roc_string, stability_bool)
    """
    poles = np.asarray(poles, dtype=np.complex128)
    if poles.size == 0:
        return "ROC: All z (no poles)", True

    mags = np.abs(poles)
    tol = 1e-9

    has_origin_pole = np.any(mags <= tol)
    has_unit_pole = np.any(np.isclose(mags, 1.0, atol=1e-6))

    if signal_type == "causal":
        r = np.max(mags)
        if has_unit_pole:
            roc = f"ROC: |z| > {r:.6f}  (pole on unit circle — ROC excludes unit circle)"
        else:
            roc = f"ROC: |z| > {r:.6f}"
        stable = np.all(mags < 1 - 1e-9)
        return roc, stable

    elif signal_type == "anti-causal":
        if has_origin_pole:
            roc = "ROC: None (pole at origin — no finite region |z| < r exists)"
            stable = np.all(mags > 1 + 1e-9)
            return roc, stable

        r = np.min(mags)
        if has_unit_pole:
            roc = f"ROC: |z| < {r:.6f}  (pole on unit circle — ROC excludes unit circle)"
        else:
            roc = f"ROC: |z| < {r:.6f}"
        stable = np.all(mags > 1 + 1e-9)
        return roc, stable

    elif signal_type == "two-sided":
        inside = mags[mags < 1 - 1e-9]
        outside = mags[mags > 1 + 1e-9]

        if inside.size == 0 and outside.size == 0:
            return "ROC: None (poles on unit circle — no two-sided ROC)", False

        r_inner = np.max(inside) if inside.size > 0 else 0.0
        r_outer = np.min(outside) if outside.size > 0 else np.inf

        if np.isinf(r_outer):
            roc = f"ROC: {r_inner:.6f} < |z| < ∞"
        else:
            roc = f"ROC: {r_inner:.6f} < |z| < {r_outer:.6f}"

        stable = (r_inner < 1.0 < r_outer)
        return roc, stable

    else:
        return f"ROC: unknown (signal_type='{signal_type}')", False


def z_plane(b, a, signal_type):
    b = np.atleast_1d(b)
    a = np.atleast_1d(a)

    zeros = np.roots(b) if b.size > 1 else np.array([])
    poles = np.roots(a) if a.size > 1 else np.array([])

    roc, stable_bool = compute_roc(poles, signal_type)
    stability_msg = "System is Stable ✅" if stable_bool else "System is Unstable ❌"

    # ----------------------
    # Determine ROC radii
    # ----------------------
    mags = np.abs(poles) if poles.size > 0 else np.array([])

    r_inner, r_outer = None, None

    if signal_type == "causal":
        r_inner = np.max(mags) if mags.size > 0 else 0
        r_outer = None                      # extends to ∞

    elif signal_type == "anti-causal":
        if mags.size == 0:
            r_outer = 0
        else:
            r_outer = np.min(mags)
        r_inner = 0                         # ROC from 0 → r_outer

    elif signal_type == "two-sided":
        inside = mags[mags < 1 - 1e-9]
        outside = mags[mags > 1 + 1e-9]

        r_inner = np.max(inside) if inside.size>0 else 0
        r_outer = np.min(outside) if outside.size>0 else None

    # ----------------------
    # Start plotting
    # ----------------------
    fig, ax = plt.subplots(figsize=(6, 6))

    # Unit circle
    unit_circle = patches.Circle((0, 0), radius=1, fill=False,
                                 edgecolor='black', linestyle='--', alpha=0.6)
    ax.add_patch(unit_circle)

    # -------- ROC SHADING --------
    shade_color = (0.4, 0.7, 1.0, 0.2)   # light blue transparency

    # Causal: |z| > r_inner
    if signal_type == "causal":
        ann = patches.Wedge((0,0),        # center
                            r=10,         # big radius (shades entire outside)
                            theta1=0, theta2=360,
                            width=max(10 - r_inner, 0.001),
                            facecolor=shade_color)
        ax.add_patch(ann)

    # Anti-causal: |z| < r_outer
    elif signal_type == "anti-causal" and r_outer is not None:
        circ = patches.Circle((0,0), r_outer,
                              facecolor=shade_color,
                              edgecolor=None)
        ax.add_patch(circ)

    # Two-sided: r_inner < |z| < r_outer
    elif signal_type == "two-sided" and r_outer is not None:
        ann = patches.Wedge((0,0),
                            r=r_outer,
                            theta1=0, theta2=360,
                            width=max(r_outer - r_inner, 0.001),
                            facecolor=shade_color)
        ax.add_patch(ann)

    # ---------------------------------------
    # Plot poles and zeros
    # ---------------------------------------
    if zeros.size>0:
        ax.plot(zeros.real, zeros.imag, 'go', markersize=10, label='Zeros')
    if poles.size>0:
        ax.plot(poles.real, poles.imag, 'rx', markersize=12, label='Poles')

    ax.axvline(0, color='gray', linewidth=0.5)
    ax.axhline(0, color='gray', linewidth=0.5)
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.set_aspect('equal')

    # Set plot limits
    pts = np.concatenate((zeros, poles)) if zeros.size + poles.size>0 else np.array([0])
    R = max(2, np.max(np.abs(pts))*1.3)
    ax.set_xlim([-R, R])
    ax.set_ylim([-R, R])

    plt.title("Z-Transform Pole–Zero Plot (with ROC shading)")
    plt.xlabel("Real Part")
    plt.ylabel("Imaginary Part")
    ax.legend()

    # Text box
    ax.text(0.02, 0.02, f"{roc}\n{stability_msg}",
            transform=ax.transAxes, fontsize=9, ha='left', va='bottom',
            bbox=dict(boxstyle="round,pad=0.3", alpha=0.15))

    # Convert to HTML
    plot_html = mpld3.fig_to_html(fig)
    plt.close(fig)

    return plot_html, roc, stability_msg


def parse_coeff_string(s):
    """
    Parse a user-provided coefficient string.
    Accepts space- or comma-separated floats. Returns a list of floats.
    """
    s = s.strip()
    if not s:
        return []
    try:
        parts = s.replace(',', ' ').split()
        return [float(x) for x in parts]
    except ValueError:
        # If parsing fails, return empty list (caller should handle)
        return []


@app.route('/z_plane1', methods=['GET', 'POST'])
def z_plane1():
    plot_html, roc, stability_msg = None, "", ""
    # Default example values
    default_num = "1"
    default_den = "1 -0.5"
    default_type = "causal"

    if request.method == 'POST':
        numerator = request.form.get('numerator', default_num)
        denominator = request.form.get('denominator', default_den)
        signal_type = request.form.get('signal_type', default_type).lower()

        b = parse_coeff_string(numerator)
        a = parse_coeff_string(denominator)

        if len(a) == 0:
            # invalid denominator — avoid crash
            roc = "Invalid denominator coefficients"
            stability_msg = "System is Unstable ❌"
            plot_html = None
        else:
            plot_html, roc, stability_msg = z_plane(b, a, signal_type)

    return render_template("z_plane1.html",
                           plot_html=plot_html,
                           roc=roc,
                           stability_msg=stability_msg,
                           default_num=default_num,
                           default_den=default_den,
                           default_type=default_type)

#################################################################################
def fourier_transfer(b, a):
    """Generates Fourier Transfer Magnitude and Phase plot as an mpld3 HTML string."""
    # Frequency range (0 to π for normalized digital frequencies)
    w = np.linspace(0, np.pi, 1024)
    jw = np.exp(1j * w)

    # Evaluate transfer function H(e^jw) = B(e^jw) / A(e^jw)
    num = np.polyval(b, jw**-1)   # numerator polynomial
    den = np.polyval(a, jw**-1)   # denominator polynomial
    H = num / den

    magnitude = np.abs(H)
    phase = np.angle(H)

    # Plot
    fig, axs = plt.subplots(2, 1, figsize=(7, 6))

    # Magnitude plot
    axs[0].plot(w, magnitude, 'b')
    axs[0].set_title("Magnitude Response |H(e^jw)|")
    axs[0].set_xlabel("Frequency (rad/sample)")
    axs[0].set_ylabel("Magnitude")
    axs[0].grid(True, linestyle=':', alpha=0.6)

    # Phase plot
    axs[1].plot(w, phase, 'r')
    axs[1].set_title("Phase Response ∠H(e^jw)")
    axs[1].set_xlabel("Frequency (rad/sample)")
    axs[1].set_ylabel("Phase (radians)")
    axs[1].grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()

    # Convert to mpld3 HTML
    plot_html = mpld3.fig_to_html(fig)
    plt.close(fig)
    return plot_html


@app.route('/fourier_transform_graph', methods=['GET', 'POST'])
def fourier_transfer_route():
    plot_html = None
    if request.method == 'POST':
        # Get coefficients from user
        numerator = request.form['numerator']
        denominator = request.form['denominator']

        # Convert input into list of floats
        b = [float(x) for x in numerator.replace(',', ' ').split()]
        a = [float(x) for x in denominator.replace(',', ' ').split()]

        plot_html = fourier_transfer(b, a)

    return render_template("fourier_transform_graph.html", plot_html=plot_html)
##################################################################################
def laplace_transfer(b, a):
    """Generates Laplace Transfer Magnitude and Phase plot as an mpld3 HTML string."""
    # Frequency range for s = jω (continuous-time)
    w = np.linspace(0, 100, 2000)  # rad/sec (adjust upper limit if needed)
    s = 1j * w

    # Evaluate transfer function H(s) = B(s) / A(s)
    num = np.polyval(b, s)
    den = np.polyval(a, s)
    H = num / den

    magnitude = np.abs(H)
    phase = np.angle(H)

    # Plot
    fig, axs = plt.subplots(2, 1, figsize=(7, 6))

    # Magnitude plot (log scale, like Bode plot)
    axs[0].semilogx(w, 20 * np.log10(magnitude), 'b')
    axs[0].set_title("Magnitude Response |H(jω)| (dB)")
    axs[0].set_xlabel("Frequency (rad/sec)")
    axs[0].set_ylabel("Magnitude (dB)")
    axs[0].grid(True, which="both", linestyle=':', alpha=0.6)

    # Phase plot
    axs[1].semilogx(w, np.degrees(phase), 'r')
    axs[1].set_title("Phase Response ∠H(jω)")
    axs[1].set_xlabel("Frequency (rad/sec)")
    axs[1].set_ylabel("Phase (degrees)")
    axs[1].grid(True, which="both", linestyle=':', alpha=0.6)

    plt.tight_layout()

    # Convert to mpld3 HTML
    plot_html = mpld3.fig_to_html(fig)
    plt.close(fig)
    return plot_html


# ----------------- SYMPY LAPLACE FUNCTION -----------------

@app.route("/laplace_transform_graph", methods=["GET", "POST"])
def laplace_transform_graph():
    import numpy as np
    import sympy as sp
    import matplotlib
    matplotlib.use("Agg")  # ensure non-GUI backend
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
    from flask import request, render_template
    import io, base64, traceback

    result = ""
    plot_html = ""

    try:
        expr_str = (request.form.get("expr") or "").strip() or "exp(-t)*sin(t)"

        def safe_get(name, default):
            try:
                return float(request.form.get(name, default))
            except Exception:
                return float(default)

        sig_min = safe_get("sig_min", "-5")
        sig_max = safe_get("sig_max", "5")
        w_min   = safe_get("w_min", "-10")
        w_max   = safe_get("w_max", "10")

        t = sp.symbols("t", real=True, positive=True)
        s = sp.symbols("s")
        alpha, omega = sp.symbols("alpha omega", real=True)

        allowed = {
            "t": t, "sin": sp.sin, "cos": sp.cos, "exp": sp.exp,
            "Heaviside": sp.Heaviside, "DiracDelta": sp.DiracDelta
        }
        expr = sp.sympify(expr_str, locals=allowed)
        F_s = sp.simplify(sp.laplace_transform(expr, t, s, noconds=True))

        result += f"<pre>x(t) = {sp.simplify(expr)}</pre>"
        result += f"<pre>F(s) = {F_s}</pre>"

        F_sub = F_s.subs(s, alpha + sp.I*omega)

        import numpy as np
        def _np_heaviside(x):  # for any Heaviside that survives into numeric form
            return np.heaviside(x, 0.5)

        F_fun = sp.lambdify((alpha, omega), F_sub, modules=[{"Heaviside": _np_heaviside}, "numpy"])
        if not callable(F_fun):
            raise RuntimeError("Internal: generated numeric function is not callable")

        N = 150
        sigma_vals = np.linspace(sig_min, sig_max, N)
        omega_vals = np.linspace(w_min, w_max, N)
        Sigma, Omega = np.meshgrid(sigma_vals, omega_vals)

        Zc = F_fun(Sigma, Omega)
        Z = np.abs(np.asarray(Zc, dtype=np.complex128))
        Z[~np.isfinite(Z)] = np.nan

        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection="3d")
        ax.plot_surface(Sigma, Omega, Z, rstride=2, cstride=2)
        ax.set_xlabel("σ (Real part of s)")
        ax.set_ylabel("ω (Imag part of s)")
        ax.set_zlabel("|F(s)|")
        ax.set_title("3D Laplace Transform Surface")

        # --- Render to PNG and embed as <img> ---
        buf = io.BytesIO()
        FigureCanvas(fig).print_png(buf)
        plt.close(fig)
        buf.seek(0)
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        plot_html = f'<img src="data:image/png;base64,{b64}" alt="Laplace Surface" />'

    except Exception as e:
        tb = traceback.format_exc()
        result = (
            f"<p style='color:red'><b>Error:</b> {e}</p>"
            f"<details><summary>Traceback</summary><pre>{tb}</pre></details>"
        )

    return render_template("laplace_transform_graph.html", result=result, plot_html=plot_html)


######################################################################
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from circuit_models.netlist_parser import parse_netlist

app.secret_key = "supersecretkey"

UPLOAD_FOLDER = "uploads"
PLOT_FOLDER = "static/plots"
ALLOWED_EXTS = {"cir", "sp", "txt"}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PLOT_FOLDER, exist_ok=True)

# dev-only "session"
SESSION = {"components": [], "file": None}

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTS


@app.route("/netlist", methods=["GET", "POST"])
def netlist_home():
    """Upload and parse netlist"""
    if request.method == "POST":
        if "netlist" not in request.files:
            flash("No file uploaded", "error")
            return redirect(request.url)

        file = request.files["netlist"]
        if not file or file.filename == "":
            flash("No file selected", "error")
            return redirect(request.url)

        if not allowed_file(file.filename):
            flash("Invalid file format (.cir, .sp, .txt allowed)", "error")
            return redirect(request.url)

        # secure the filename
        fname = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, fname)
        file.save(filepath)

        try:
            comps = parse_netlist(filepath)
            SESSION["components"] = comps or []
            SESSION["file"] = fname
            flash(f"Parsed {len(SESSION['components'])} component(s) from {fname}", "success")
            return redirect(url_for("netlist_home"))
        except Exception as e:
            tb = traceback.format_exc()
            flash(f"Error parsing netlist: {e}\n{tb}", "error")
            return redirect(request.url)

    return render_template(
        "netlist.html",
        components=SESSION["components"],
        file_token=SESSION["file"],
        plot_image=None,
        selected=None,
    )


@app.route("/netlist/plot/<int:idx>")
def netlist_plot(idx):
    """Plot I-V curve of a component"""
    try:
        comp = SESSION["components"][idx]
    except (IndexError, KeyError, TypeError):
        flash("Component not found", "error")
        return redirect(url_for("netlist_home"))

    if not hasattr(comp, "plot_iv") or not callable(getattr(comp, "plot_iv")):
        flash(f"{getattr(comp, 'name', 'Component')} does not support plotting", "warning")
        return redirect(url_for("netlist_home"))

    try:
        # unique filename to avoid stale browser cache/overwrite
        base = getattr(comp, "name", f"comp_{idx}")
        plot_file = f"{secure_filename(str(base))}_{uuid.uuid4().hex[:8]}.png"
        plot_path = os.path.join(PLOT_FOLDER, plot_file)

        # fresh figure, let component draw onto current axes
        fig = plt.figure(figsize=(6, 4))
        ax = fig.add_subplot(111)

        # comp.plot_iv() should plot on current axes; if it returns fig/ax, you can adapt
        comp.plot_iv()  # ensure this function does not call plt.show()

        ax.set_title(f"I–V: {base}")
        ax.grid(True, alpha=0.3)

        fig.tight_layout()
        fig.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close(fig)  # very important to avoid file locks on Windows

        return render_template(
            "netlist.html",
            components=SESSION["components"],
            file_token=SESSION["file"],
            plot_image=f"plots/{plot_file}",  # in template: url_for('static', filename=plot_image)
            selected=base,
        )
    except Exception as e:
        tb = traceback.format_exc()
        flash(f"Error plotting: {e}\n{tb}", "error")
        return redirect(url_for("netlist_home"))
    
# --- extra imports near top of app.py ---
import uuid, io, base64
import networkx as nx
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from circuit_models.netlist_parser import build_netgraph, dc_operating_point

# Utility to save current fig to static/plots and return relative path
def _save_current_plot(basename: str) -> str:
    from matplotlib import pyplot as plt
    plot_file = f"{basename}_{uuid.uuid4().hex[:8]}.png"
    plot_path = os.path.join(PLOT_FOLDER, plot_file)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    return f"plots/{plot_file}"

@app.route("/netlist/plot_ac/<int:idx>")
def netlist_plot_ac(idx):
    try:
        comp = SESSION["components"][idx]
    except IndexError:
        flash("Component not found", "error")
        return redirect(url_for("netlist_home"))

    if comp.ctype not in ("C","L") or not hasattr(comp, "plot_ac"):
        flash(f"{getattr(comp,'name','Component')} does not support AC plot", "warning")
        return redirect(url_for("netlist_home"))

    try:
        plt.clf()
        comp.plot_ac()
        rel = _save_current_plot(f"{comp.name}_ac")
        return render_template(
            "netlist.html",
            components=SESSION["components"],
            file_token=SESSION["file"],
            plot_image=rel,
            selected=f"{comp.name} (AC)",
            schematic_image=None,
            dc_solution=None,
        )
    except Exception as e:
        flash(f"Error plotting AC: {e}\n{traceback.format_exc()}", "error")
        return redirect(url_for("netlist_home"))

@app.route("/netlist/schematic")
def netlist_schematic():
    comps = SESSION.get("components") or []
    if not comps:
        flash("No components loaded", "warning")
        return redirect(url_for("netlist_home"))
    try:
        nodes, edges = build_netgraph(comps)
        G = nx.Graph()
        for n in nodes: G.add_node(n)
        for n1, n2, label in edges: G.add_edge(n1, n2, label=label)

        plt.clf()
        fig = plt.figure(figsize=(7,5))
        ax = fig.add_subplot(111)
        pos = nx.spring_layout(G, seed=42)
        nx.draw(G, pos, with_labels=True, node_color="#eef3ff", edge_color="#999", node_size=900, ax=ax)
        nx.draw_networkx_edge_labels(G, pos, edge_labels={(u,v):d['label'] for u,v,d in G.edges(data=True)}, font_size=8, ax=ax)
        ax.set_axis_off()
        rel = _save_current_plot("schematic")

        return render_template(
            "netlist.html",
            components=comps,
            file_token=SESSION["file"],
            plot_image=None,
            selected=None,
            schematic_image=rel,
            dc_solution=None,
        )
    except Exception as e:
        flash(f"Error drawing schematic: {e}\n{traceback.format_exc()}", "error")
        return redirect(url_for("netlist_home"))

@app.route("/netlist/dc")
def netlist_dc():
    comps = SESSION.get("components") or []
    if not comps:
        flash("No components loaded", "warning")
        return redirect(url_for("netlist_home"))
    try:
        sol = dc_operating_point(comps)
        if not sol:
            flash("DC solver found no solvable nodes (need R and/or DC sources).", "warning")
        return render_template(
            "netlist.html",
            components=comps,
            file_token=SESSION["file"],
            plot_image=None,
            selected=None,
            schematic_image=None,
            dc_solution=sol or None,
        )
    except Exception as e:
        flash(f"Error in DC OP: {e}\n{traceback.format_exc()}", "error")
        return redirect(url_for("netlist_home"))

########################################################################################
@app.route("/emft", methods=["GET", "POST"])
def emft():
    result = ""
    plot_html = ""

    if request.method == "POST":
        try:
            # Replace ^ with **
            def normalize(expr):
                return (expr or "").replace("^", "**")

            Fx_str = normalize(request.form.get("Fx"))
            Fy_str = normalize(request.form.get("Fy"))
            Fz_str = normalize(request.form.get("Fz"))

            # SymPy expressions
            x, y, z = sp.symbols('x y z')
            Fx = sp.sympify(Fx_str)
            Fy = sp.sympify(Fy_str)
            Fz = sp.sympify(Fz_str)

            # Parse point
            pt = [float(v) for v in request.form.get("point", "0,0,0").split(",")]
            while len(pt) < 3:  
                pt.append(0.0)

            # Divergence
            divF = sp.diff(Fx, x) + sp.diff(Fy, y) + sp.diff(Fz, z)

            # Curl
            c1 = sp.diff(Fz, y) - sp.diff(Fy, z)
            c2 = sp.diff(Fx, z) - sp.diff(Fz, x)
            c3 = sp.diff(Fy, x) - sp.diff(Fx, y)

            result += "<h3>Symbolic Results</h3>"
            result += f"<pre>Divergence:\n{sp.pretty(divF)}</pre>"
            result += f"<pre>Curl:\n({sp.pretty(c1)}, {sp.pretty(c2)}, {sp.pretty(c3)})</pre>"

            # Evaluate at point
            subs = {x: pt[0], y: pt[1], z: pt[2]}
            div_val = divF.subs(subs)
            c1_val = c1.subs(subs)
            c2_val = c2.subs(subs)
            c3_val = c3.subs(subs)

            result += "<h3>Values at Point</h3>"
            result += f"<pre>Divergence at {pt}: {div_val}</pre>"
            result += f"<pre>Curl at {pt}: ({c1_val}, {c2_val}, {c3_val})</pre>"

            # ----------- IN-MEMORY GRAPH (NO SAVE) -----------

            # xy-grid around point
            span = float(request.form.get("span", 2))
            xs = np.linspace(pt[0] - span, pt[0] + span, 12)
            ys = np.linspace(pt[1] - span, pt[1] + span, 12)
            X, Y = np.meshgrid(xs, ys)

            Fx_num = sp.lambdify((x, y, z), Fx, "numpy")
            Fy_num = sp.lambdify((x, y, z), Fy, "numpy")

            U = Fx_num(X, Y, pt[2])
            V = Fy_num(X, Y, pt[2])

            if np.isscalar(U): U = np.full_like(X, U)
            if np.isscalar(V): V = np.full_like(Y, V)

            plt.figure(figsize=(5, 4))
            plt.quiver(X, Y, U, V)
            plt.title(f"Vector Field (z={pt[2]})")
            plt.xlabel("x")
            plt.ylabel("y")
            plt.tight_layout()

            # Convert to base64
            buf = io.BytesIO()
            plt.savefig(buf, format="png")
            plt.close()
            buf.seek(0)
            img_base64 = base64.b64encode(buf.read()).decode("utf-8")

            plot_html = f"""
            <h3>Graph</h3>
            <img src="data:image/png;base64,{img_base64}" 
                 style="max-width:100%; border:1px solid #ccc;">
            """

        except Exception as e:
            result = f"<p style='color:red'>Error: {e}</p>"

    return render_template("emft.html", result=result + plot_html)
################################################################################
@app.route("/coords", methods=["GET", "POST"])
def coords():
    result = ""
    plot_html = ""

    def parse_floats(s):
        # parse comma separated floats, ignore empty entries
        parts = [p.strip() for p in (s or "").split(",") if p.strip() != ""]
        vals = []
        for p in parts:
            try:
                vals.append(float(p))
            except:
                # try sympy evaluation (for pi, etc.)
                try:
                    vals.append(float(sp.N(sp.sympify(p))))
                except Exception:
                    raise ValueError(f"Can't parse number: '{p}'")
        return vals

    # conversion helpers
    def cart2polar(x, y, angle_in_degrees=False):
        r = np.hypot(x, y)
        theta = np.arctan2(y, x)  # radians
        if angle_in_degrees:
            theta = np.degrees(theta)
        return (r, theta)

    def polar2cart(r, theta, angle_in_degrees=False):
        if angle_in_degrees:
            theta = np.radians(theta)
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        return (x, y)

    def cart2cyl(x, y, z, angle_in_degrees=False):
        r, theta = cart2polar(x, y, angle_in_degrees=False)
        if angle_in_degrees:
            theta = np.degrees(theta)
        return (r, theta, z)

    def cyl2cart(r, theta, z, angle_in_degrees=False):
        if angle_in_degrees:
            theta = np.radians(theta)
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        return (x, y, z)

    def cart2sph(x, y, z, angle_in_degrees=False):
        # using physics convention: (rho, theta, phi)
        # where rho = distance, theta = azimuth (from x axis), phi = inclination (from +z axis)
        rho = np.sqrt(x*x + y*y + z*z)
        theta = np.arctan2(y, x)           # azimuth (0..2pi) around z
        # handle rho==0 to avoid division by zero
        phi = 0.0
        if rho != 0:
            phi = np.arccos(z / rho)      # inclination 0..pi
        if angle_in_degrees:
            theta = np.degrees(theta)
            phi = np.degrees(phi)
        return (rho, theta, phi)

    def sph2cart(rho, theta, phi, angle_in_degrees=False):
        # theta = azimuth, phi = inclination from +z
        if angle_in_degrees:
            theta = np.radians(theta)
            phi = np.radians(phi)
        x = rho * np.sin(phi) * np.cos(theta)
        y = rho * np.sin(phi) * np.sin(theta)
        z = rho * np.cos(phi)
        return (x, y, z)

    try:
        if request.method == "POST":
            # inputs
            src = request.form.get("system_from", "cartesian").strip().lower()
            dst = request.form.get("system_to", "polar").strip().lower()
            coords_str = request.form.get("coords", "")
            angle_unit = request.form.get("angle_unit", "rad").strip().lower()
            plot_flag = request.form.get("plot", "off") == "on"

            angle_in_degrees = (angle_unit == "deg")

            vals = parse_floats(coords_str)

            # Normalize accepted names to a few canonical ones
            # Accept: cartesian / rectangular / rect
            #         polar / circular (2D)
            #         cylindrical / cyl (3D)
            #         spherical / sph (3D)
            alias = {
                "rectangular": "cartesian", "rect": "cartesian",
                "cartesian": "cartesian", "cart": "cartesian",
                "polar": "polar", "circular": "polar",
                "cylindrical": "cylindrical", "cyl": "cylindrical",
                "spherical": "spherical", "sph": "spherical",
            }
            src = alias.get(src, src)
            dst = alias.get(dst, dst)

            # Validate coordinate length by system type
            def required_len(system):
                if system == "polar": return 2
                if system == "cartesian": return 2  # default to 2D unless user supplies 3
                if system == "cylindrical": return 3
                if system == "spherical": return 3
                return None

            # If user supplied 3 numbers but asked cartesian <-> polar, we treat appropriately:
            # If 2D requested, ignore extra z.
            # If 3 numbers and system is cartesian, assume 3D cartesian (x,y,z)
            # So we'll branch by the actual provided values and the chosen src/dst.

            # Branch conversions:
            conv_result = None
            conv_detail = ""

            # --- CARTESIAN -> ...
            if src == "cartesian":
                if len(vals) == 0:
                    raise ValueError("No coordinate values provided.")
                if len(vals) == 1:
                    # one value: treat as x, 1D linear coordinate
                    x = vals[0]
                    # simple identity or to polar not meaningful
                    if dst in ("cartesian",):
                        conv_result = (x,)
                        conv_detail = f"1D linear coordinate: x = {x}"
                    else:
                        raise ValueError("1D linear coordinate cannot be converted to the requested 2D/3D system.")
                elif len(vals) == 2:
                    x, y = vals
                    if dst == "polar":
                        r, theta = cart2polar(x, y, angle_in_degrees=angle_in_degrees)
                        conv_result = (float(r), float(theta))
                        conv_detail = f"Polar (r, θ): ({r:.6g}, {theta:.6g}{'°' if angle_in_degrees else ' rad'})"
                    elif dst == "cartesian":
                        conv_result = (float(x), float(y))
                        conv_detail = f"Cartesian (x, y): ({x}, {y})"
                    elif dst == "cylindrical":
                        # promote to 3D with z=0
                        r, theta, z = cart2cyl(x, y, 0.0, angle_in_degrees=angle_in_degrees)
                        conv_result = (float(r), float(theta), float(z))
                        conv_detail = f"Cylindrical (r, θ, z): ({r:.6g}, {theta:.6g}{'°' if angle_in_degrees else ' rad'}, {z})"
                    elif dst == "spherical":
                        # promote to 3D with z=0
                        rho, theta, phi = cart2sph(x, y, 0.0, angle_in_degrees=angle_in_degrees)
                        conv_result = (float(rho), float(theta), float(phi))
                        conv_detail = f"Spherical (ρ, θ, φ): ({rho:.6g}, {theta:.6g}{'°' if angle_in_degrees else ' rad'}, {phi:.6g}{'°' if angle_in_degrees else ' rad'})"
                    else:
                        raise ValueError(f"Unsupported target system: {dst}")
                else:
                    # 3 values: treat as 3D cartesian (x,y,z)
                    x, y, z = vals[:3]
                    if dst == "cartesian":
                        conv_result = (float(x), float(y), float(z))
                        conv_detail = f"Cartesian (x, y, z): ({x}, {y}, {z})"
                    elif dst == "cylindrical":
                        r, theta, zz = cart2cyl(x, y, z, angle_in_degrees=angle_in_degrees)
                        conv_result = (float(r), float(theta), float(zz))
                        conv_detail = f"Cylindrical (r, θ, z): ({r:.6g}, {theta:.6g}{'°' if angle_in_degrees else ' rad'}, {zz})"
                    elif dst == "spherical":
                        rho, theta, phi = cart2sph(x, y, z, angle_in_degrees=angle_in_degrees)
                        conv_result = (float(rho), float(theta), float(phi))
                        conv_detail = f"Spherical (ρ, θ, φ): ({rho:.6g}, {theta:.6g}{'°' if angle_in_degrees else ' rad'}, {phi:.6g}{'°' if angle_in_degrees else ' rad'})"
                    elif dst == "polar":
                        # drop z
                        r, theta = cart2polar(x, y, angle_in_degrees=angle_in_degrees)
                        conv_result = (float(r), float(theta))
                        conv_detail = f"Polar (r, θ) (z dropped): ({r:.6g}, {theta:.6g}{'°' if angle_in_degrees else ' rad'})"
                    else:
                        raise ValueError(f"Unsupported target system: {dst}")

            # --- POLAR -> ...
            elif src == "polar":
                if len(vals) < 2:
                    raise ValueError("Polar requires 2 values (r, θ).")
                r = vals[0]; theta = vals[1]
                if dst == "cartesian":
                    x, y = polar2cart(r, theta, angle_in_degrees=angle_in_degrees)
                    conv_result = (float(x), float(y))
                    conv_detail = f"Cartesian (x,y): ({x:.6g}, {y:.6g})"
                elif dst == "polar":
                    conv_result = (float(r), float(theta))
                    conv_detail = f"Polar (r, θ): ({r}, {theta}{'°' if angle_in_degrees else ' rad'})"
                elif dst == "cylindrical":
                    # promote z optional (3rd value)
                    z = vals[2] if len(vals) >= 3 else 0.0
                    x, y = polar2cart(r, theta, angle_in_degrees=angle_in_degrees)
                    R, Th, Z = cart2cyl(x, y, z, angle_in_degrees=angle_in_degrees)
                    conv_result = (float(R), float(Th), float(Z))
                    conv_detail = f"Cylindrical (r, θ, z): ({R:.6g}, {Th:.6g}{'°' if angle_in_degrees else ' rad'}, {Z})"
                elif dst == "spherical":
                    z = vals[2] if len(vals) >= 3 else 0.0
                    x, y = polar2cart(r, theta, angle_in_degrees=angle_in_degrees)
                    rho, th, phi = cart2sph(x, y, z, angle_in_degrees=angle_in_degrees)
                    conv_result = (float(rho), float(th), float(phi))
                    conv_detail = f"Spherical (ρ, θ, φ): ({rho:.6g}, {th:.6g}{'°' if angle_in_degrees else ' rad'}, {phi:.6g}{'°' if angle_in_degrees else ' rad'})"
                else:
                    raise ValueError(f"Unsupported target system: {dst}")

            # --- CYLINDRICAL -> ...
            elif src == "cylindrical":
                if len(vals) < 3:
                    raise ValueError("Cylindrical requires 3 values (r, θ, z).")
                r = vals[0]; theta = vals[1]; z = vals[2]
                if dst == "cartesian":
                    x, y, zz = cyl2cart(r, theta, z, angle_in_degrees=angle_in_degrees)
                    conv_result = (float(x), float(y), float(zz))
                    conv_detail = f"Cartesian (x, y, z): ({x:.6g}, {y:.6g}, {zz})"
                elif dst == "cylindrical":
                    conv_result = (float(r), float(theta), float(z))
                    conv_detail = f"Cylindrical (r, θ, z): ({r}, {theta}{'°' if angle_in_degrees else ' rad'}, {z})"
                elif dst == "spherical":
                    x, y, zz = cyl2cart(r, theta, z, angle_in_degrees=angle_in_degrees)
                    rho, th, phi = cart2sph(x, y, zz, angle_in_degrees=angle_in_degrees)
                    conv_result = (float(rho), float(th), float(phi))
                    conv_detail = f"Spherical (ρ, θ, φ): ({rho:.6g}, {th:.6g}{'°' if angle_in_degrees else ' rad'}, {phi:.6g}{'°' if angle_in_degrees else ' rad'})"
                elif dst == "polar":
                    # drop z and return polar for r,theta
                    conv_result = (float(r), float(theta))
                    conv_detail = f"Polar (r, θ) (z dropped): ({r}, {theta}{'°' if angle_in_degrees else ' rad'})"
                else:
                    raise ValueError(f"Unsupported target system: {dst}")

            # --- SPHERICAL -> ...
            elif src == "spherical":
                if len(vals) < 3:
                    raise ValueError("Spherical requires 3 values (ρ, θ, φ).")
                rho = vals[0]; theta = vals[1]; phi = vals[2]
                if dst == "cartesian":
                    x, y, z = sph2cart(rho, theta, phi, angle_in_degrees=angle_in_degrees)
                    conv_result = (float(x), float(y), float(z))
                    conv_detail = f"Cartesian (x, y, z): ({x:.6g}, {y:.6g}, {z:.6g})"
                elif dst == "spherical":
                    conv_result = (float(rho), float(theta), float(phi))
                    conv_detail = f"Spherical (ρ, θ, φ): ({rho}, {theta}{'°' if angle_in_degrees else ' rad'}, {phi}{'°' if angle_in_degrees else ' rad'})"
                elif dst == "cylindrical":
                    x, y, z = sph2cart(rho, theta, phi, angle_in_degrees=angle_in_degrees)
                    r, th, zz = cart2cyl(x, y, z, angle_in_degrees=angle_in_degrees)
                    conv_result = (float(r), float(th), float(zz))
                    conv_detail = f"Cylindrical (r, θ, z): ({r:.6g}, {th:.6g}{'°' if angle_in_degrees else ' rad'}, {zz})"
                elif dst == "polar":
                    # project to xy-plane and return polar (r,theta)
                    x, y, z = sph2cart(rho, theta, phi, angle_in_degrees=angle_in_degrees)
                    r, th = cart2polar(x, y, angle_in_degrees=angle_in_degrees)
                    conv_result = (float(r), float(th))
                    conv_detail = f"Polar (r, θ): ({r:.6g}, {th:.6g}{'°' if angle_in_degrees else ' rad'})"
                else:
                    raise ValueError(f"Unsupported target system: {dst}")

            else:
                raise ValueError(f"Unsupported source system: {src}")

            # Format result HTML
            result += "<h3>Conversion Result</h3>"
            result += f"<pre>From: {src}  →  To: {dst}\nInput: {vals}\nResult: {conv_result}</pre>"
            result += f"<p>{conv_detail}</p>"

            # Optional 2D plot for conversions involving 2D points (cartesian <-> polar)
            if plot_flag:
                make_plot = False
                px = py = None
                if src == "polar" and dst == "cartesian":
                    # show both input (in polar) and output (in cartesian)
                    r_in, th_in = vals[0], vals[1]
                    x_out, y_out = polar2cart(r_in, th_in, angle_in_degrees=angle_in_degrees)
                    px, py = x_out, y_out
                    make_plot = True
                elif src == "cartesian" and dst in ("polar", "cylindrical", "spherical"):
                    if len(vals) >= 2:
                        px, py = vals[0], vals[1]
                        make_plot = True
                elif src in ("cylindrical",) and dst == "cartesian":
                    px, py = cyl2cart(vals[0], vals[1], vals[2], angle_in_degrees=angle_in_degrees)[:2]
                    make_plot = True
                elif src == "spherical" and dst == "cartesian":
                    # project to xy plane for visualization
                    x, y, z = sph2cart(vals[0], vals[1], vals[2], angle_in_degrees=angle_in_degrees)
                    px, py = x, y
                    make_plot = True

                if make_plot and px is not None and py is not None:
                    plt.figure(figsize=(4.5,4))
                    # draw axes and point
                    plt.axhline(0, linewidth=0.6)
                    plt.axvline(0, linewidth=0.6)
                    plt.scatter([px], [py])
                    plt.text(px, py, f"  ({px:.4g}, {py:.4g})")
                    plt.title("Projection in xy-plane")
                    plt.xlabel("x")
                    plt.ylabel("y")
                    plt.gca().set_aspect('equal', adjustable='box')
                    plt.tight_layout()
                    buf = io.BytesIO()
                    plt.savefig(buf, format="png")
                    plt.close()
                    buf.seek(0)
                    img_base64 = base64.b64encode(buf.read()).decode("utf-8")
                    plot_html = f"""
                    <h3>Plot (xy projection)</h3>
                    <img src="data:image/png;base64,{img_base64}" style="max-width:100%; border:1px solid #ccc;">
                    """

    except Exception as e:
        result = f"<p style='color:red'>Error: {e}</p>"

    return render_template("coords.html", result=result + plot_html)

################################################################################
@app.route("/inverse_fourier", methods=["GET", "POST"])
def inverse_fourier():
    import sympy as sp
    from flask import request, render_template
    import traceback

    result = ""

    try:
        # ----- Input -----
        # X(ω) in SymPy syntax, default is a classic lowpass rectangle: 1 for |ω|<1
        Xw_str = (request.form.get("Xw") or "").strip() or "Piecewise((1, Abs(w) < 1), (0, True))"

        # ----- Symbols -----
        t = sp.symbols("t", real=True)     # time variable
        w = sp.symbols("w", real=True)     # frequency variable (rad/s)

        # Allow only safe names for sympify
        allowed = {
            "w": w, "t": t,
            "pi": sp.pi,
            "I": sp.I,
            "Abs": sp.Abs,
            "sign": sp.sign,
            "signum": sp.sign,
            "Heaviside": sp.Heaviside,
            "DiracDelta": sp.DiracDelta,
            "Piecewise": sp.Piecewise,
            "sin": sp.sin, "cos": sp.cos, "exp": sp.exp,
            "sqrt": sp.sqrt, "log": sp.log, "atan": sp.atan,
            "oo": sp.oo
        }

        # Parse X(ω)
        Xw = sp.sympify(Xw_str, locals=allowed)

        # ----- Inverse Fourier Transform (SymPy's default convention) -----
        xt = sp.inverse_fourier_transform(Xw, w, t)  # returns symbolic x(t)

        # Optional tidy-up
        xt_simpl = sp.simplify(xt)

        # ----- Echo + result -----
        result += "<h3>Inverse Fourier Transform</h3>"
        result += f"<pre>X(ω) = {sp.srepr(Xw) if isinstance(Xw, sp.Basic) and Xw.is_Atom else Xw}</pre>"
        result += f"<pre>x(t) = {xt_simpl}</pre>"
        result += "<p><small>Convention: F{x(t)}=∫ x(t)e^{-jωt}dt, inverse=(1/2π)∫ X(ω)e^{jωt}dω.</small></p>"

    except Exception as e:
        tb = traceback.format_exc()
        result = (
            f"<p style='color:red'><b>Error:</b> {e}</p>"
            f"<details><summary>Traceback</summary><pre>{tb}</pre></details>"
        )

    # Your template should render {{ result|safe }}
    return render_template("inverse_fourier.html", result=result)

################################################################################
@app.route("/fresnel", methods=["GET", "POST"])
def fresnel():
    result = ""
    plot_html = ""
    ray_plot_html = ""
    try:
        if request.method == "POST":
            # Get user input
            eps1 = float(sp.sympify(request.form.get("eps1")))
            mu1  = float(sp.sympify(request.form.get("mu1")))
            eps2 = float(sp.sympify(request.form.get("eps2")))
            mu2  = float(sp.sympify(request.form.get("mu2")))
            E0   = float(sp.sympify(request.form.get("E0")))
            theta_i_deg = float(request.form.get("theta"))

            # Convert angle
            theta_i = np.radians(theta_i_deg)

            # Intrinsic impedances
            eta1 = np.sqrt(mu1/eps1)
            eta2 = np.sqrt(mu2/eps2)

            # Snell's law: sin(theta_t) = (n1/n2) * sin(theta_i)
            n1 = np.sqrt(mu1 * eps1)
            n2 = np.sqrt(mu2 * eps2)
            sin_theta_t = (n1/n2) * np.sin(theta_i)
            
            # Determine which medium is denser (higher refractive index)
            if n2 > n1:
                dense_side = "right"
            else:
                dense_side = "left"

            # Handle Total internal reflection
            if abs(sin_theta_t) > 1:
                result = "<p style='color:red'>Total Internal Reflection occurs! (No transmission)</p>"
                theta_t = np.nan
            else:
                theta_t = np.arcsin(sin_theta_t)

            # Fresnel coefficients
            # TE (s polarization)
            Gamma_TE = (eta2*np.cos(theta_i) - eta1*np.cos(theta_t)) / \
                       (eta2*np.cos(theta_i) + eta1*np.cos(theta_t))
            T_TE = (2*eta2*np.cos(theta_i)) / \
                   (eta2*np.cos(theta_i) + eta1*np.cos(theta_t))

            # TM (p polarization)
            Gamma_TM = (eta1*np.cos(theta_i) - eta2*np.cos(theta_t)) / \
                       (eta1*np.cos(theta_i) + eta2*np.cos(theta_t))
            T_TM = (2*eta2*np.cos(theta_i)) / \
                   (eta1*np.cos(theta_i) + eta2*np.cos(theta_t))

            # --------- Fresnel Ray Diagram Plot ---------
            # --------- Fresnel Ray Diagram Plot ---------
            fig2, ax2 = plt.subplots(figsize=(6,6))

            # Medium boundary (vertical line at x=0)
            ax2.plot([0,0],[-1,1], linewidth=2)

            # Normal (dashed)
            ax2.plot([-1,1],[0,0], linestyle="--")

            # Ray origin
            x0, y0 = -0.8, 0

            # Incident ray
            xi = x0 + np.cos(theta_i)
            yi = y0 + np.sin(theta_i)
            ax2.arrow(x0, y0, xi-x0, yi-y0, head_width=0.05, length_includes_head=True)

            # Reflected ray
            xr = x0 + np.cos(theta_i)
            yr = y0 - np.sin(theta_i)
            ax2.arrow(x0, y0, xr-x0, yr-y0, head_width=0.05, length_includes_head=True)

            # Transmitted ray (only if not TIR)
            if not np.isnan(theta_t):
                xt = 0.8*np.cos(theta_t)
                yt = -0.8*np.sin(theta_t)
                ax2.arrow(0,0, xt, yt, head_width=0.05, length_includes_head=True)

            # Labels
            ax2.text(-0.7,0.6,"Incident Wave", fontsize=10)
            ax2.text(-0.7,-0.6,"Reflected Wave", fontsize=10)
            ax2.text(0.3,-0.6,"Transmitted Wave", fontsize=10)
            ax2.text(-0.15,0.05,"Normal", fontsize=9)

            # Medium label
            ax2.text(-0.9,0.9,"Medium 1", fontsize=12)
            ax2.text(0.3,0.9,"Medium 2", fontsize=12)

            # Axis settings
            ax2.set_xlim(-1,1)
            ax2.set_ylim(-1,1)
            ax2.set_title("Fresnel Reflection & Refraction Geometry")
            ax2.set_aspect('equal')
            ax2.axis("off")

            # Background shading based on medium density
            if dense_side == "left":
                ax2.add_patch(plt.Rectangle((-1, -1), 1, 2, alpha=0.25))  # left darker
                ax2.add_patch(plt.Rectangle((0, -1), 1, 2, alpha=0.05))   # right lighter
            else:
                ax2.add_patch(plt.Rectangle((0, -1), 1, 2, alpha=0.25))   # right darker
                ax2.add_patch(plt.Rectangle((-1, -1), 1, 2, alpha=0.05))  # left lighter

            ray_plot_html = mpld3.fig_to_html(fig2)
            plt.close(fig2)


            # Time axis
            t = np.linspace(0, 20, 500)
            omega = 1.0

            Ei = E0 * np.cos(omega*t)
            Er_TE = Gamma_TE * E0 * np.cos(omega*t)
            Et_TE = T_TE * E0 * np.cos(omega*t)
            Er_TM = Gamma_TM * E0 * np.cos(omega*t)
            Et_TM = T_TM * E0 * np.cos(omega*t)

            # Store Results
            result += f"<pre>Incident Angle: {theta_i_deg:.2f}°</pre>"
            if not np.isnan(theta_t):
                result += f"<pre>Transmitted Angle: {np.degrees(theta_t):.2f}°</pre><br>"

            result += "<b>TE (s-pol)</b>"
            result += f"<pre>Γ_TE = {Gamma_TE:.4f}   T_TE = {T_TE:.4f}</pre>"
            result += "<b>TM (p-pol)</b>"
            result += f"<pre>Γ_TM = {Gamma_TM:.4f}   T_TM = {T_TM:.4f}</pre>"

            # Plot
            fig, ax = plt.subplots(figsize=(7,4))
            ax.plot(t, Ei, label="Incident")
            ax.plot(t, Er_TE, "--", label="Reflected TE")
            ax.plot(t, Et_TE, "--", label="Transmitted TE")
            ax.plot(t, Er_TM, "-.", label="Reflected TM")
            ax.plot(t, Et_TM, "-.", label="Transmitted TM")
            ax.set_xlabel("Time")
            ax.set_ylabel("Amplitude")
            ax.set_title("Fresnel Reflection & Transmission (Oblique Incidence)")
            ax.legend()

            plot_html = mpld3.fig_to_html(fig)
            plt.close(fig)

    except Exception as e:
        result = f"<p style='color:red'>Error: {e}</p>"

    return render_template("fresnel.html", result=result,
                       plot_html=plot_html,
                       ray_plot_html=ray_plot_html)
#####################################################################################


@app.route("/smith", methods=["GET", "POST"])
def smith():
    result = ""
    plot_html = ""

    import numpy as np
    import matplotlib.pyplot as plt
    import sympy as sp
    import skrf as rf
    import mpld3

    try:
        # Defaults if first load
        Z0 = float(request.form.get("Z0", 50))
        RL = float(request.form.get("RL", 100))
        XL = float(request.form.get("XL", 50))
        theta_deg = float(request.form.get("theta", 0))  # slider value

        ZL = RL + 1j * XL

        # Reflection coefficient
        Gamma = (ZL - Z0) / (ZL + Z0)
        mag_Gamma = abs(Gamma)
        phase_Gamma = np.angle(Gamma, deg=True)
        VSWR = (1 + mag_Gamma) / (1 - mag_Gamma) if mag_Gamma < 1 else np.inf
        ReturnLoss = -20 * np.log10(mag_Gamma) if mag_Gamma != 0 else np.inf

        Z_norm = ZL / Z0
        result += f"<pre>ZL = {ZL:.2f} Ω</pre>"
        result += f"<pre>Γ = {Gamma:.3f} | |Γ| = {mag_Gamma:.3f} @ {phase_Gamma:.2f}°</pre>"
        result += f"<pre>VSWR = {VSWR:.3f}</pre>"
        result += f"<pre>Return Loss = {ReturnLoss:.2f} dB</pre>"
        result += f"<pre>z = {Z_norm:.3f}</pre>"
        
        # --- L-section MATCHING ---
        # Case: RL < Z0 → series L, shunt C (high-Q)
        # Case: RL > Z0 → series C, shunt L (low-Q)

        if RL < Z0:
            Q = np.sqrt(Z0 / RL - 1)
            Xs = Q * Z0      # series reactance
            Xp = RL * Q      # shunt reactance
            match_type = "Series Inductor, Shunt Capacitor (RL < Z0)"
        else:
            Q = np.sqrt(RL / Z0 - 1)
            Xs = Z0 / Q      # series C
            Xp = RL / Q      # shunt L
            match_type = "Series Capacitor, Shunt Inductor (RL > Z0)"

        result += f"<pre>L-Match: {match_type}\nSeries Xs = {Xs:.2f} Ω, Shunt Xp = {Xp:.2f} Ω</pre>"

        # Convert matching impedance path to Gamma points
        def Z_to_Gamma(Z): return (Z - Z0) / (Z + Z0)

        # Series reactance path
        Z_series = ZL + 1j * np.linspace(0, Xs, 200)
        gamma_series = Z_to_Gamma(Z_series)

        # Shunt reactance path (move in admittance plane)
        YL = 1 / ZL
        Y_shunt = YL + 1j * np.linspace(0, 1/Xp, 200)
        Z_shunt = 1 / Y_shunt
        gamma_shunt = Z_to_Gamma(Z_shunt)

        # --- Smith chart ---
        fig, ax = plt.subplots(figsize=(8, 8))
        rf.plotting.smith(ax=ax, chart_type="z", draw_labels=True)

        # Load point
        ax.plot(np.real(Gamma), np.imag(Gamma), 'o', markersize=10, label="Load Γ")

        # VSWR circle
        th = np.linspace(0, 2*np.pi, 400)
        ax.plot(mag_Gamma * np.cos(th), mag_Gamma * np.sin(th), "--", linewidth=1)

        # L-match paths
        ax.plot(np.real(gamma_series), np.imag(gamma_series), label="Series X", linewidth=2)
        ax.plot(np.real(gamma_shunt), np.imag(gamma_shunt), label="Shunt X", linewidth=2)

        # TL rotation arc (slider)
        theta = np.deg2rad(theta_deg)
        rot_trace = Gamma * np.exp(-1j * 2 * np.linspace(0, theta, 200))
        ax.plot(np.real(rot_trace), np.imag(rot_trace), linewidth=2, label=f"TL Rotation {theta_deg}°")

        # Rotated point
        Gamma_rot = Gamma * np.exp(-1j * 2 * theta)
        ax.plot(np.real(Gamma_rot), np.imag(Gamma_rot), 's', markersize=8, label="Rotated Γ")

        ax.set_xlim(-1.05, 1.05)
        ax.set_ylim(-1.05, 1.05)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title("Enhanced Smith Chart with L-Match and TL Rotation")
        ax.legend(loc="upper right", fontsize=8)

        plot_html = mpld3.fig_to_html(fig)
        plt.close(fig)

    except Exception as e:
        result = f"<p style='color:red'>Error: {e}</p>"

    return render_template("smith.html", result=result, plot_html=plot_html)


##################################################################################

@app.route("/waveguide", methods=["GET", "POST"])
def waveguide():
    result = ""
    plot_html = ""

    try:
        if request.method == "POST":

            import numpy as np
            import matplotlib.pyplot as plt
            import mpld3
            from scipy.special import jv, jn_zeros, jvp

            # User selections
            wg_type = request.form.get("wg_type", "rectangular")
            field_type = request.form.get("field_type", "E")

            c = 3e8
            mu0 = 4 * np.pi * 1e-7
            eps0 = 1 / (mu0 * c * c)

            # Mode indices
            m = int(request.form.get("m"))
            n = int(request.form.get("n"))

            # Prevent invalid mode
            if m == 0 and n == 0:
                raise ValueError("TM(0,0) mode does not physically exist.")

            # =============== RECTANGULAR WAVEGUIDE ===============
            if wg_type == "rectangular":
                a = float(request.form.get("a"))
                b = float(request.form.get("b"))

                fc = (c / 2) * np.sqrt((m / a) ** 2 + (n / b) ** 2)

                x = np.linspace(0, a, 200)
                y = np.linspace(0, b, 200)
                X, Y = np.meshgrid(x, y)

                # TM Electric field
                Ez = np.sin(m * np.pi * X / a) * np.sin(n * np.pi * Y / b)

                # Magnetic fields (from ∇Ez)
                dEz_dx = (m * np.pi / a) * np.cos(m * np.pi * X / a) * np.sin(n * np.pi * Y / b)
                dEz_dy = (n * np.pi / b) * np.sin(m * np.pi * X / a) * np.cos(n * np.pi * Y / b)

                Hx = -(1j / mu0) * dEz_dy
                Hy =  (1j / mu0) * dEz_dx

                # Correct magnitude (complex magnitude)
                H_mag = np.abs(np.sqrt(Hx**2 + Hy**2))

                field = Ez if field_type == "E" else H_mag
                title = f"Rectangular TM({m},{n}) - {'Ez' if field_type=='E' else '|H|'}"

            # =============== CIRCULAR WAVEGUIDE ===============
            else:
                radius = float(request.form.get("radius"))

                roots = jn_zeros(m, n)
                k_mn = roots[-1] / radius   # TM mode root

                fc = (c / (2*np.pi)) * k_mn

                r = np.linspace(0, radius, 200)
                theta = np.linspace(0, 2*np.pi, 200)
                R, T = np.meshgrid(r, theta)

                # TM Electric field
                Ez = jv(m, k_mn * R) * np.cos(m * T)

                # Correct derivative dEz/dr = k * Jm'(kr)
                dEz_dr = k_mn * jvp(m, k_mn * R, n=1)

                # dEz / dθ
                dEz_dθ = -m * jv(m, k_mn * R) * np.sin(m * T)

                # Avoid division by zero at r = 0
                denom = R + 1e-12
                H_mag = np.abs(np.sqrt((dEz_dr)**2 + (dEz_dθ/denom)**2))

                field = Ez if field_type == "E" else H_mag
                title = f"Circular TM({m},{n}) - {'Ez' if field_type=='E' else '|H|'}"

                # Convert polar to Cartesian for plotting
                X = R * np.cos(T)
                Y = R * np.sin(T)

            # ===== Plotting =====
            fig, ax = plt.subplots(figsize=(6,5))
            img = ax.pcolormesh(X, Y, field, shading="auto", cmap="jet")
            fig.colorbar(img, ax=ax)
            ax.set_aspect("equal", adjustable="box")
            ax.set_title(title)
            ax.set_xlabel("x (m)")
            ax.set_ylabel("y (m)")

            plot_html = mpld3.fig_to_html(fig)
            plt.close(fig)

            result += f"<pre>Cutoff Frequency: {fc/1e9:.3f} GHz</pre>"

    except Exception as e:
        result = f"<p style='color:red'>Error: {e}</p>"

    return render_template("waveguide.html", result=result, plot_html=plot_html)

###############################################################################################
@app.route("/bounce", methods=["GET", "POST"])
def bounce():
    result = ""
    plot_html = ""

    # helper to try multiple possible keys and parse float safely
    def get_float_from_form(possible_keys, default):
        for k in possible_keys:
            v = request.form.get(k)
            if v is None:
                continue
            try:
                return float(v)
            except Exception:
                # try trimming and replacing commas
                try:
                    return float(v.strip().replace(",", ""))
                except Exception:
                    continue
        return default

    try:
        if request.method == "POST":
            mode = (request.form.get("mode") or "tdr").lower()  # 'tdr' or 'two_wall'
            # DEBUG: log the incoming form contents so you can confirm names/values

            import numpy as np
            import matplotlib.pyplot as plt
            import mpld3

            td = get_float_from_form(["td", "time_delay", "td_ns"], 1.0)
            n_reflections = int(get_float_from_form(["n_reflections", "nref", "n"], 6))

            # ------------------ TDR (unchanged) ------------------
            if mode == "tdr":
                Z0 = get_float_from_form(["Z0","z0"], 50.0)
                ZL = get_float_from_form(["ZL","zL","Z_L","Z_Load"], 75.0)
                Zs = get_float_from_form(["Zs","zs","Zs_src"], 50.0)
                Vstep = get_float_from_form(["Vstep","V_step","V0"], 1.0)
                observe = request.form.get("observe", "load")

                Gamma_L = (ZL - Z0) / (ZL + Z0)
                Gamma_S = (Zs - Z0) / (Zs + Z0)
                V0_plus = Vstep * Z0 / (Zs + Z0)

                times = [0]
                voltages = [V0_plus if observe == "load" else Vstep - V0_plus]

                Vcur = V0_plus
                t = 0

                for k in range(1, n_reflections + 1):
                    # alternate L then S after initial arrival
                    Vcur *= (Gamma_L if k % 2 == 1 else Gamma_S)
                    t += td
                    times.append(t)
                    voltages.append(voltages[-1] + Vcur)

                fig, ax = plt.subplots(figsize=(7, 4))
                ax.step(times, voltages, where="post")
                ax.set_xlabel("Time (ns)")
                ax.set_ylabel("Voltage (V)")
                ax.set_title("TDR Bounce Plot")
                ax.grid(True)

                result += f"<pre>ΓL={Gamma_L:.3f}, ΓS={Gamma_S:.3f}</pre>"

            # ------------------ TWO-WALL Bounce (robust) ------------------
            else:
                # Try many possible field names to avoid mismatch bugs
                V0 = get_float_from_form(["V0","V_initial","V"], 1.0)
                Gamma_L = get_float_from_form(
                    ["Gamma_L","GammaL","gamma_l","gammaL","gL"], 0.5
                )
                Gamma_R = get_float_from_form(
                    ["Gamma_R","GammaR","gamma_r","gammaR","gR"], -0.3
                )

                # Accept start_side names too
                start_side_raw = (request.form.get("start_side") or
                                  request.form.get("startSide") or
                                  request.form.get("start", "left")).lower()
                start_side = "right" if "right" in start_side_raw else "left"

                # DEBUG: log what values we resolved
                # start position: 0=left, 1=right
                current_side = 1 if start_side == "right" else 0

                x = [current_side]
                y = [0]
                amplitudes = [V0]

                V = V0
                for k in range(1, n_reflections + 1):
                    # toggle side explicitly
                    current_side = 1 - current_side
                    x.append(current_side)
                    y.append(k)

                    # apply proper coefficient based on the wall hit
                    if current_side == 0:
                        V = V * Gamma_L
                    else:
                        V = V * Gamma_R

                    amplitudes.append(V)

                fig, ax = plt.subplots(figsize=(5, 8))
                ax.plot(x, y, marker='o', linestyle='-')

                for i, amp in enumerate(amplitudes):
                    xpos = x[i]
                    x_offset = 0.06 if xpos == 0 else -0.18
                    ax.text(
                        xpos + x_offset,
                        y[i],
                        f"{amp:.3f} V",
                        fontsize=9,
                        verticalalignment='center'
                    )

                ax.set_xticks([0, 1])
                ax.set_xticklabels([f"Left (ΓL={Gamma_L:.3f})", f"Right (ΓR={Gamma_R:.3f})"])
                ax.set_ylabel("Bounce index (k)")
                ax.set_xlabel("Wall position")
                ax.set_title("Two-Wall Voltage Bounce Diagram")
                ax.grid(True, linestyle="--", alpha=0.4)
                ax.set_xlim(-0.2, 1.2)

                result += f"<pre>Start side: {start_side.capitalize()}, Γ_left={Gamma_L:.3f}, Γ_right={Gamma_R:.3f}</pre>"

            # ------------------ Convert to HTML ------------------
            plot_html = mpld3.fig_to_html(fig)
            plt.close(fig)
            result += f"<pre>Reflections simulated: {n_reflections}</pre>"

    except Exception as e:
        # show error to user and log
        result = f"<p style='color:red'>Error: {e}</p>"

    return render_template("bounce.html", result=result, plot_html=plot_html)


# ---------------- CLIPPER CIRCUIT ----------------
def diode_clipper(input_signal, E1, E2, Vk):
    output_signal = input_signal.copy()
    for i in range(len(output_signal)):
        if output_signal[i] > E1 + Vk:
            output_signal[i] = E1 + Vk
        elif output_signal[i] < -(E2 + Vk):
            output_signal[i] = -(E2 + Vk)
    return output_signal

def simulate_half_wave(Vp, Vk, R, f):
    t = np.linspace(0, 1/f, 500)
    Vin = Vp * np.sin(2*np.pi*f*t)
    Vout = np.maximum(Vin - Vk, 0)
    fig, ax = plt.subplots()
    ax.plot(t, Vout)
    ax.set_title("Half-Wave Rectifier Output")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Voltage (V)")
    return mpld3.fig_to_html(fig)

def simulate_full_wave(Vp, Vk, R, f):
    t = np.linspace(0, 1/f, 500)
    Vin = Vp * np.sin(2*np.pi*f*t)
    Vout = np.abs(Vin) - Vk
    Vout[Vout < 0] = 0
    fig, ax = plt.subplots()
    ax.plot(t, Vout)
    ax.set_title("Full-Wave Rectifier Output")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Voltage (V)")
    return mpld3.fig_to_html(fig)

def simulate_rc(Vin, R, C, f):
    t = np.linspace(0, 1/f, 500)
    Vin_wave = Vin*np.sin(2*np.pi*f*t)
    tau = R*C
    Vout = Vin_wave/(np.sqrt(1+(2*np.pi*f*tau)**2))
    fig, ax = plt.subplots()
    ax.plot(t, Vout)
    ax.set_title("RC Low-Pass Filter Output")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Voltage (V)")
    return mpld3.fig_to_html(fig)

def simulate_rl(Vin, R, L, f):
    t = np.linspace(0, 1/f, 500)
    Vin_wave = Vin*np.sin(2*np.pi*f*t)
    Xl = 2*np.pi*f*L
    I = Vin_wave/np.sqrt(R**2 + Xl**2)
    fig, ax = plt.subplots()
    ax.plot(t, I)
    ax.set_title("RL Circuit Current")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Current (A)")
    return mpld3.fig_to_html(fig)

def simulate_rlc(Vin, R, L, C, f):
    t = np.linspace(0, 1/f, 500)
    Vin_wave = Vin*np.sin(2*np.pi*f*t)
    w = 2*np.pi*f
    Xl = w*L
    Xc = 1/(w*C)
    Z = np.sqrt(R**2 + (Xl-Xc)**2)
    I = Vin_wave / Z
    fig, ax = plt.subplots()
    ax.plot(t, I)
    ax.set_title("RLC Band-Pass Circuit Output Current")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Current (A)")
    return mpld3.fig_to_html(fig)

@app.route("/clipper", methods=["GET","POST"])
def clipper():
    plot = None
    circuit = request.form.get("circuit")

    if request.method == "POST" and circuit:
        if circuit == "half":
            plot = simulate_half_wave(float(request.form["Vp"]), float(request.form["Vk"]),
                                      float(request.form["Rl"]), float(request.form["freq"]))
        elif circuit == "full":
            plot = simulate_full_wave(float(request.form["Vp"]), float(request.form["Vk"]),
                                      float(request.form["Rl"]), float(request.form["freq"]))
        elif circuit == "rc":
            plot = simulate_rc(float(request.form["Vin"]), float(request.form["R"]),
                               float(request.form["C"]), float(request.form["freq"]))
        elif circuit == "rl":
            plot = simulate_rl(float(request.form["Vin"]), float(request.form["R"]),
                               float(request.form["L"]), float(request.form["freq"]))
        elif circuit == "rlc":
            plot = simulate_rlc(float(request.form["Vin"]), float(request.form["R"]),
                                float(request.form["L"]), float(request.form["C"]),
                                float(request.form["freq"]))

    return render_template("clipper.html", plot=plot, circuit=circuit)

######################################################################################################
@app.route('/DTFT', methods=['GET', 'POST'])
def DTFT():
    """
    DTFT / inverse DTFT calculator using Lcapy (route name uses uppercase /DTFT).
    Same style & behavior as your z_transform route; template name unchanged: 'dtft.html'.
    Form fields:
      - signal_expr : expression of x[n] or X(w)
      - operation   : 'dtft' or 'idtft'
    """
    results = []
    error = None

    examples = {
        'dtft_examples': [
            'u(n)',
            '0.9**n * u(n)',
            '(-0.5)**n * u(n)',
            'cos(0.5*pi*n) * u(n)',
            'sin(0.25*pi*n) * u(n)'
        ],
        'idtft_examples': [
            '1 / (1 - 0.9*exp(-j*w))',
            '1 / (1 - 0.5*exp(-j*w))',
            'exp(-j*w) + exp(-j*2*w)'
        ]
    }

    try:
        # Prefer lcapy.expr if available (same parsing behavior as your z_transform)
        from lcapy import expr
        from lcapy import symbols as lcapy_symbols
        HAVE_LCAPY = True
    except Exception:
        HAVE_LCAPY = False
        # Minimal fallback: expose a thin expr wrapper using sympy if needed
        def expr(s):
            # lazy import to keep startup light
            from lcapy import expr as _expr  # if lcapy absent this won't run
            return _expr(s)

    if request.method == 'POST':
        try:
            user_input = request.form.get("signal_expr", "").strip()
            if not user_input:
                raise ValueError("Input cannot be empty.")

            operation = request.form.get("operation", "dtft").lower()

            # Parse using lcapy.expr → supports n, w, u(n), exp(-j*w), etc.
            E = expr(user_input)

            # Show parsed input
            results.append({
                'description': "Input (parsed)",
                'expr': str(E),
                'latex': getattr(E, 'latex', lambda: str(E))()
            })

            # ==========================
            #       FORWARD DTFT
            # ==========================
            if operation == "dtft":
                try:
                    # Use Lcapy's built-in DTFT operator
                    Xw = E.DTFT()
                except Exception:
                    # fallback — some expressions use call form with w
                    from lcapy import symbols
                    w = symbols('w', real=True)
                    Xw = E(w)

                # Simplify aggressively
                Xw_simpl = getattr(Xw, "simplify", lambda: Xw)()

                results.append({
                    'description': "DTFT X(w)",
                    'expr': str(Xw_simpl),
                    'latex': getattr(Xw_simpl, 'latex', lambda: str(Xw_simpl))()
                })

            # ==========================
            #       INVERSE DTFT
            # ==========================
            elif operation == "idtft":
                try:
                    xn = E.IDTFT()
                except Exception:
                    # fallback: call with n
                    from lcapy import symbols
                    n = symbols('n', integer=True)
                    xn = E(n)

                xn_simpl = getattr(xn, "simplify", lambda: xn)()

                results.append({
                    'description': "Inverse DTFT (sequence x[n])",
                    'expr': str(xn_simpl),
                    'latex': getattr(xn_simpl, 'latex', lambda: str(xn_simpl))()
                })

            else:
                raise ValueError("Unknown operation. Use 'dtft' or 'idtft'.")

        except Exception as e:
            import traceback
            error = f"{e}\n{traceback.format_exc()}"

    return render_template("DTFT.html",
                           error=error,
                           results=results,
                           examples=examples)

################################################################################################

def bit_reverse_indices(n):
    bits = int(math.log2(n))
    return np.array([int('{:0{w}b}'.format(i, w=bits)[::-1], 2) for i in range(n)])

def staged_fft_dit(x):
    """Return list of numpy arrays: stage0..stageK for DIT radix-2 FFT.
       stage0 = input after bit-reversal (so viewers see permutation first).
    """
    x = np.asarray(x, dtype=complex)
    N = x.size
    assert N & (N - 1) == 0, "N must be power of two"
    stages = []
    # bit-reverse input ordering (so stage0 = permuted inputs)
    br = bit_reverse_indices(N)
    state = x[br].copy()
    stages.append(state.copy())

    half_size = 1
    n_stages = int(math.log2(N))
    for stage in range(1, n_stages + 1):
        step = half_size * 2
        W_m = np.exp(-2j * np.pi / step)
        for k in range(0, N, step):
            w = 1.0 + 0j
            for j in range(half_size):
                a = state[k + j]
                b = state[k + j + half_size] * w
                state[k + j] = a + b
                state[k + j + half_size] = a - b
                w *= W_m
        stages.append(state.copy())
        half_size = step
    return stages

def svg_tree_from_stages(stages, width=1000, height_per_node=38, node_r=18):
    """
    Build an SVG string showing stages as columns. Each column shows N nodes.
    All nodes are present at all stages; node labels update to stage values.
    We include connector lines (butterfly cross) and mark which pairs are
    combined at each stage by data attributes for JS to highlight.
    """
    N = stages[0].size
    levels = len(stages)  # stage0..stageK
    canvas_width = width
    canvas_height = max(300, height_per_node * N)

    left_margin = 80
    right_margin = 60
    top_margin = 40
    bottom_margin = 40
    usable_w = canvas_width - left_margin - right_margin
    col_x = lambda s: left_margin + s * (usable_w / max(1, (levels - 1)))

    def node_y(idx):
        # distribute from top_margin..canvas_height-bottom_margin
        top = top_margin
        bottom = canvas_height - bottom_margin
        if N == 1:
            return (top + bottom) / 2
        return top + (bottom - top) * (idx / (N - 1))

    svg = []
    svg.append(f'<svg id="fft-svg" viewBox="0 0 {canvas_width} {canvas_height}" '
               f'xmlns="http://www.w3.org/2000/svg" style="background:#fff;">')

    # Stage labels
    for s in range(levels):
        x = col_x(s)
        svg.append(f'<text x="{x}" y="{top_margin/2}" font-size="14" text-anchor="middle" fill="#111">Stage {s}</text>')

    # Precompute position dict
    pos = {}
    for s in range(levels):
        x = col_x(s)
        for i in range(N):
            y = node_y(i)
            pos[(s, i)] = (x, y)

    # Draw connector lines and mark butterflies as groups
    for s in range(levels - 1):
        step = 2 ** (s)
        # group id for stage s
        for i0 in range(0, N, 2 * step):
            for j in range(step):
                i = i0 + j
                jidx = i + step
                p_a = pos[(s, i)]
                p_b = pos[(s, jidx)]
                p_a2 = pos[(s + 1, i)]
                p_b2 = pos[(s + 1, jidx)]
                # straight lines down/up
                # connections a->a2 and b->b2
                svg.append(f'<line x1="{p_a[0]+node_r}" y1="{p_a[1]}" x2="{p_a2[0]-node_r}" y2="{p_a2[1]}" stroke="#9aa5b1" stroke-width="2" />')
                svg.append(f'<line x1="{p_b[0]+node_r}" y1="{p_b[1]}" x2="{p_b2[0]-node_r}" y2="{p_b2[1]}" stroke="#9aa5b1" stroke-width="2" />')
                # cross lines for butterfly
                # a -> b2 and b -> a2 (dashed)
                svg.append(f'<line class="butterfly-line stage-{s}" data-stage="{s}" x1="{p_a[0]+node_r}" y1="{p_a[1]}" x2="{p_b2[0]-node_r}" y2="{p_b2[1]}" stroke="#cbd5e1" stroke-width="1.5" stroke-dasharray="4 4" />')
                svg.append(f'<line class="butterfly-line stage-{s}" data-stage="{s}" x1="{p_b[0]+node_r}" y1="{p_b[1]}" x2="{p_a2[0]-node_r}" y2="{p_a2[1]}" stroke="#cbd5e1" stroke-width="1.5" stroke-dasharray="4 4" />')

    # Draw nodes (circles + text). Each node group contains label placeholders for JS updates.
    for s in range(levels):
        for i in range(N):
            x, y = pos[(s, i)]
            idg = f"n{s}_{i}"
            # initial value from stages[s][i]
            val = stages[s][i]
            val_str = f"{val.real:.2f}" if abs(val.imag) < 1e-8 else f"{val.real:.2f}{val.imag:+.2f}j"
            # group with data-stage and data-idx
            svg.append(f'<g id="{idg}" class="node" data-stage="{s}" data-idx="{i}">')
            svg.append(f'  <circle cx="{x}" cy="{y}" r="{node_r}" fill="#ffffff" stroke="#1f2937" stroke-width="1.5"/>')
            svg.append(f'  <text x="{x}" y="{y-6}" font-size="10" text-anchor="middle" fill="#111">{i}</text>')
            svg.append(f'  <text class="val" x="{x}" y="{y+10}" font-size="10" text-anchor="middle"  font-weight="bold" fill="#0b1220">{html.escape(val_str)}</text>')
            svg.append('</g>')

    svg.append('</svg>')
    return '\n'.join(svg)


@app.route("/fft", methods=["GET", "POST"])
def fft():
    final_output = []
    svg_tree = None
    N_choices = [2,4,8,16]
    error = None
    chosen_N = 8
    if request.method == "POST":
        raw = request.form.get("input_values","").strip()
        chosen_N = int(request.form.get("fft_points","8"))
        if chosen_N not in N_choices:
            error = "Select FFT size 2,4,8 or 16."
        else:
            # parse inputs
            try:
                parts = [p.strip() for p in raw.split(",") if p.strip()!='']
                vals = []
                for p in parts:
                    # allow complex like 1+2j or '3' floats
                    try:
                        v = complex(p)
                    except:
                        v = complex(float(p), 0.0)
                    vals.append(v)
                # pad or trim to chosen_N
                if len(vals) < chosen_N:
                    vals += [0]*(chosen_N - len(vals))
                else:
                    vals = vals[:chosen_N]
                stages = staged_fft_dit(vals)
                svg_tree = svg_tree_from_stages(stages)
                final_vals = stages[-1]
                for i, v in enumerate(final_vals):
                    val_str = f"{v.real:.2f}" if abs(v.imag) < 1e-8 else f"{v.real:.2f}{v.imag:+.2f}j"
                    final_output.append((i, val_str))
            except Exception as e:
                error = "Failed to parse inputs. Use comma-separated numbers, complex allowed like 1+2j."
    else:
        # default example
        chosen_N = 8
        vals = [1.0 * np.sin(2*np.pi*1*k/chosen_N) + 0.5*np.sin(2*np.pi*2*k/chosen_N) for k in range(chosen_N)]
        stages = staged_fft_dit(vals)
        svg_tree = svg_tree_from_stages(stages)

    return render_template("fft.html", svg_tree=svg_tree, error=error, chosen_N=chosen_N, final_output=final_output)
###################################################################################################
# cpu8085.py
# Simple 8085 CPU emulator (educational, not cycle-accurate).
# Supports a practical subset of instructions: MOV, MVI, LXI, LDA, STA,
# LHLD/SHLD, ADD/ADI, SUB/SUI, INR/DCR, INX/DCX, JMP, JZ/JNZ, JC/JNC,
# CALL/RET, PUSH/POP, HLT, NOP, CMP/CPI, ANI/ORI/XRA/ANA, OUT/IN, RLC/RRC, CMA.
# Extend opcode table and behavior as needed.

from typing import List, Tuple, Dict

REG_ORDER = ["B","C","D","E","H","L","M","A"]  # for encoding convenience

def word(high, low):
    return ((high & 0xFF) << 8) | (low & 0xFF)

class CPU8085:
    def __init__(self, mem_size=65536):
        self.reg = {r:0 for r in ["A","B","C","D","E","H","L"]}
        self.flags = {"Z":0, "S":0, "P":0, "CY":0, "AC":0}
        self.PC = 0
        self.SP = 0xFFFE
        self.memory = [0]*mem_size
        self.halted = False
        self.output_log = []
        self.input_ports = {}
        # simple opcode dispatch table will be built later if needed

    def reset(self):
        for r in self.reg: self.reg[r]=0
        for f in self.flags: self.flags[f]=0
        self.PC = 0
        self.SP = 0xFFFE
        self.memory = [0]*len(self.memory)
        self.halted = False
        self.output_log = []

    def load_program(self, code: List[int], addr=0x0000):
        for i, b in enumerate(code):
            self.memory[addr+i] = b & 0xFF
        self.PC = addr
        self.halted = False

    def read_mem(self, addr):
        return self.memory[addr & 0xFFFF]

    def write_mem(self, addr, val):
        self.memory[addr & 0xFFFF] = val & 0xFF

    def get_reg_pair(self, rp):
        if rp == "B":
            return (self.reg["B"]<<8) | self.reg["C"]
        if rp == "D":
            return (self.reg["D"]<<8) | self.reg["E"]
        if rp == "H":
            return (self.reg["H"]<<8) | self.reg["L"]
        if rp == "PSW":
            return (self.reg["A"]<<8) | self._flags_to_byte()
        return 0

    def set_reg_pair(self, rp, val):
        val &= 0xFFFF
        if rp == "B":
            self.reg["B"] = (val>>8)&0xFF; self.reg["C"]=val&0xFF
        if rp == "D":
            self.reg["D"] = (val>>8)&0xFF; self.reg["E"]=val&0xFF
        if rp == "H":
            self.reg["H"] = (val>>8)&0xFF; self.reg["L"]=val&0xFF
        if rp == "PSW":
            self.reg["A"] = (val>>8)&0xFF; self._byte_to_flags(val&0xFF)

    def _flags_to_byte(self):
        b = 0
        b |= (self.flags["S"]<<7)
        b |= (self.flags["Z"]<<6)
        b |= (0<<5)  # x (unused)
        b |= (self.flags["AC"]<<4)
        b |= (0<<3)  # x
        b |= (self.flags["P"]<<2)
        b |= (1<<1)  # always 1 in PSW? (not exact) - keep placeholder
        b |= (self.flags["CY"]<<0)
        return b

    def _byte_to_flags(self, b):
        self.flags["S"] = (b>>7)&1
        self.flags["Z"] = (b>>6)&1
        self.flags["AC"] = (b>>4)&1
        self.flags["P"] = (b>>2)&1
        self.flags["CY"] = b&1

    def _set_flag_szp(self, value):
        v = value & 0xFF
        self.flags["Z"] = 1 if v==0 else 0
        self.flags["S"] = 1 if (v & 0x80)!=0 else 0
        # parity
        self.flags["P"] = 1 if bin(v).count("1") % 2 == 0 else 0

    def push_word(self, val):
        self.SP = (self.SP - 1) & 0xFFFF
        self.write_mem(self.SP, (val>>8)&0xFF)
        self.SP = (self.SP - 1) & 0xFFFF
        self.write_mem(self.SP, val&0xFF)

    def pop_word(self):
        low = self.read_mem(self.SP); self.SP = (self.SP + 1)&0xFFFF
        high = self.read_mem(self.SP); self.SP = (self.SP + 1)&0xFFFF
        return (high<<8)|low

    def fetch(self):
        b = self.read_mem(self.PC)
        self.PC = (self.PC+1) & 0xFFFF
        return b

    def step(self) -> Tuple[bool,str]:
        """Execute single instruction. Returns (halted, message)."""
        if self.halted:
            return True, "HALTED"

        op = self.fetch()
        # NOP
        if op == 0x00:
            return False, "NOP"

        # HLT
        if op == 0x76:
            self.halted = True
            return True, "HLT"

        # MVI r, data8 : 0x06, 0x0E, 0x16, 0x1E, 0x26, 0x2E, 0x36, 0x3E
        if op in (0x06,0x0E,0x16,0x1E,0x26,0x2E,0x36,0x3E):
            idx = (op>>3)&0x07
            reg = REG_ORDER[idx]
            data = self.fetch()
            if reg == "M":
                addr = (self.reg["H"]<<8)|self.reg["L"]
                self.write_mem(addr, data)
            else:
                self.reg[reg] = data
            return False, f"MVI {reg},#{data:02X}"

        # MOV r1,r2 : 0x40 - 0x7F (except 0x76 handled above)
        if 0x40 <= op <= 0x7F:
            dst = (op>>3)&0x07
            src = op & 0x07
            dst_r = REG_ORDER[dst]; src_r = REG_ORDER[src]
            val = None
            if src_r == "M":
                addr = (self.reg["H"]<<8)|self.reg["L"]; val = self.read_mem(addr)
            else:
                val = self.reg[src_r]
            if dst_r == "M":
                addr = (self.reg["H"]<<8)|self.reg["L"]; self.write_mem(addr, val)
            else:
                self.reg[dst_r] = val
            return False, f"MOV {dst_r},{src_r}"

        # LXI rp, word : 0x01,0x11,0x21,0x31  (B, D, H, SP)
        if op in (0x01,0x11,0x21,0x31):
            low = self.fetch(); high = self.fetch()
            val = (high<<8)|low
            if op == 0x01: self.set_reg_pair("B", val)
            elif op == 0x11: self.set_reg_pair("D", val)
            elif op == 0x21: self.set_reg_pair("H", val)
            elif op == 0x31: self.SP = val
            return False, f"LXI {op:02X} #{val:04X}"

        # LDA addr (3A), STA (32)
        if op == 0x3A:
            low = self.fetch(); high = self.fetch(); addr = (high<<8)|low
            self.reg["A"] = self.read_mem(addr)
            return False, f"LDA {addr:04X}"
        if op == 0x32:
            low = self.fetch(); high = self.fetch(); addr = (high<<8)|low
            self.write_mem(addr, self.reg["A"])
            return False, f"STA {addr:04X}"

        # LHLD (2A) / SHLD (22)
        if op == 0x2A:
            low = self.fetch(); high = self.fetch(); addr = (high<<8)|low
            self.reg["L"] = self.read_mem(addr)
            self.reg["H"] = self.read_mem((addr+1)&0xFFFF)
            return False, f"LHLD {addr:04X}"
        if op == 0x22:
            low = self.fetch(); high = self.fetch(); addr = (high<<8)|low
            self.write_mem(addr, self.reg["L"]); self.write_mem((addr+1)&0xFFFF, self.reg["H"])
            return False, f"SHLD {addr:04X}"

        # ADD r (80-87) / ADI (C6)
        if 0x80 <= op <= 0x87:
            src = op & 0x07
            r = REG_ORDER[src]
            val = self.reg[r] if r!="M" else self.read_mem((self.reg["H"]<<8)|self.reg["L"])
            res = self.reg["A"] + val
            self.flags["CY"] = 1 if res>0xFF else 0
            self.reg["A"] = res & 0xFF
            self._set_flag_szp(self.reg["A"])
            return False, f"ADD {r}"
        if op == 0xC6:
            data = self.fetch()
            res = self.reg["A"] + data
            self.flags["CY"] = 1 if res>0xFF else 0
            self.reg["A"] = res & 0xFF
            self._set_flag_szp(self.reg["A"])
            return False, f"ADI #{data:02X}"

        # SUB r (90-97) / SUI (D6)
        if 0x90 <= op <= 0x97:
            src = op & 0x07
            r = REG_ORDER[src]
            val = self.reg[r] if r!="M" else self.read_mem((self.reg["H"]<<8)|self.reg["L"])
            res = (self.reg["A"] - val) & 0x1FF
            self.flags["CY"] = 1 if self.reg["A"] < val else 0
            self.reg["A"] = res & 0xFF
            self._set_flag_szp(self.reg["A"]); return False, f"SUB {r}"
        if op == 0xD6:
            data = self.fetch()
            res = (self.reg["A"] - data) & 0x1FF
            self.flags["CY"] = 1 if self.reg["A"] < data else 0
            self.reg["A"] = res & 0xFF
            self._set_flag_szp(self.reg["A"]); return False, f"SUI #{data:02X}"

        # INR r (04,0C,14,1C,24,2C,34,3C) / DCR r (05,0D,15,...)
        if op in (0x04,0x0C,0x14,0x1C,0x24,0x2C,0x34,0x3C):
            idx = (op>>3)&0x07; r = REG_ORDER[idx]
            if r=="M":
                addr = (self.reg["H"]<<8)|self.reg["L"]; v = (self.read_mem(addr)+1)&0xFF; self.write_mem(addr, v)
                self._set_flag_szp(v)
            else:
                self.reg[r] = (self.reg[r]+1)&0xFF
                self._set_flag_szp(self.reg[r])
            return False, f"INR {r}"
        if op in (0x05,0x0D,0x15,0x1D,0x25,0x2D,0x35,0x3D):
            idx = (op>>3)&0x07; r = REG_ORDER[idx]
            if r=="M":
                addr = (self.reg["H"]<<8)|self.reg["L"]; v = (self.read_mem(addr)-1)&0xFF; self.write_mem(addr, v)
                self._set_flag_szp(v)
            else:
                self.reg[r] = (self.reg[r]-1)&0xFF
                self._set_flag_szp(self.reg[r])
            return False, f"DCR {r}"

        # INX rp (03,13,23,33) DCX rp (0B,1B,2B,3B)
        if op in (0x03,0x13,0x23,0x33):
            if op==0x03: rp="B"
            elif op==0x13: rp="D"
            elif op==0x23: rp="H"
            else: rp="SP"
            if rp=="SP":
                self.SP = (self.SP + 1)&0xFFFF
            else:
                v = self.get_reg_pair(rp)+1; self.set_reg_pair(rp, v&0xFFFF)
            return False, f"INX {rp}"
        if op in (0x0B,0x1B,0x2B,0x3B):
            if op==0x0B: rp="B"
            elif op==0x1B: rp="D"
            elif op==0x2B: rp="H"
            else: rp="SP"
            if rp=="SP":
                self.SP = (self.SP - 1)&0xFFFF
            else:
                v = (self.get_reg_pair(rp)-1)&0xFFFF; self.set_reg_pair(rp, v)
            return False, f"DCX {rp}"

        # JMP, JZ, JNZ, JC, JNC
        if op in (0xC3, 0xCA, 0xC2, 0xDA, 0xD2):
            low = self.fetch(); high = self.fetch(); addr = (high<<8)|low
            if op==0xC3:
                self.PC = addr; return False, f"JMP {addr:04X}"
            if op==0xC2:
                if self.flags["Z"]==0: self.PC = addr; return False, f"JNZ {addr:04X}"
                return False, f"JNZ (no)"
            if op==0xCA:
                if self.flags["Z"]==1: self.PC = addr; return False, f"JZ {addr:04X}"
                return False, f"JZ (no)"
            if op==0xD2:
                if self.flags["CY"]==0: self.PC = addr; return False, f"JNC {addr:04X}"
                return False, f"JNC (no)"
            if op==0xDA:
                if self.flags["CY"]==1: self.PC = addr; return False, f"JC {addr:04X}"
                return False, f"JC (no)"

        # CALL (CD) / RET (C9)
        if op == 0xCD:
            low = self.fetch(); high = self.fetch(); addr = (high<<8)|low
            ret = self.PC
            self.push_word(ret)
            self.PC = addr
            return False, f"CALL {addr:04X}"
        if op == 0xC9:
            self.PC = self.pop_word()
            return False, "RET"

        # PUSH rp (C5, D5, E5, F5) POP rp (C1, D1, E1, F1)
        if op in (0xC5,0xD5,0xE5,0xF5):
            if op==0xC5: rp="B"
            elif op==0xD5: rp="D"
            elif op==0xE5: rp="H"
            else: rp="PSW"
            if rp=="PSW":
                self.push_word((self.reg["A"]<<8) | self._flags_to_byte())
            else:
                self.push_word(self.get_reg_pair(rp))
            return False, f"PUSH {rp}"
        if op in (0xC1,0xD1,0xE1,0xF1):
            if op==0xC1: rp="B"
            elif op==0xD1: rp="D"
            elif op==0xE1: rp="H"
            else: rp="PSW"
            val = self.pop_word()
            if rp=="PSW":
                self.reg["A"] = (val>>8)&0xFF; self._byte_to_flags(val&0xFF)
            else:
                self.set_reg_pair(rp, val)
            return False, f"POP {rp}"

        # CMP r (B8-BF) CPI D? (FE)
        if 0xB8 <= op <= 0xBF:
            src = op & 0x07; r = REG_ORDER[src]
            val = self.reg[r] if r!="M" else self.read_mem((self.reg["H"]<<8)|self.reg["L"])
            res = (self.reg["A"] - val) & 0xFF
            self.flags["CY"] = 1 if self.reg["A"] < val else 0
            self._set_flag_szp(res)
            return False, f"CMP {r}"
        if op == 0xFE:
            data = self.fetch()
            res = (self.reg["A"] - data) & 0xFF
            self.flags["CY"] = 1 if self.reg["A"] < data else 0
            self._set_flag_szp(res)
            return False, f"CPI #{data:02X}"

        # ANI, ORI, XRA, ANA immediate
        if op == 0xE6: # ANI
            data = self.fetch(); self.reg["A"] &= data; self._set_flag_szp(self.reg["A"]); self.flags["CY"]=0; return False, f"ANI #{data:02X}"
        if op == 0xF6: # ORI
            data = self.fetch(); self.reg["A"] |= data; self._set_flag_szp(self.reg["A"]); self.flags["CY"]=0; return False, f"ORI #{data:02X}"
        if op == 0xEE: # XRI
            data = self.fetch(); self.reg["A"] ^= data; self._set_flag_szp(self.reg["A"]); self.flags["CY"]=0; return False, f"XRI #{data:02X}"
        if op == 0xE0: # RST / OUT/IN not covered here
            pass

        # OUT (D3) IN (DB)
        if op == 0xD3:
            port = self.fetch()
            self.output_log.append(("OUT", port, self.reg["A"]))
            return False, f"OUT #{port:02X}"
        if op == 0xDB:
            port = self.fetch()
            val = self.input_ports.get(port, 0)
            self.reg["A"] = val & 0xFF
            return False, f"IN #{port:02X}"

        # RLC (07), RRC (0F), CMA (2F)
        if op == 0x07:
            a = self.reg["A"]
            carry = (a>>7)&1
            a = ((a<<1)&0xFF) | carry
            self.reg["A"] = a; self.flags["CY"]=carry
            return False, "RLC"
        if op == 0x0F:
            a = self.reg["A"]
            carry = a&1
            a = (a>>1) | (carry<<7)
            self.reg["A"] = a; self.flags["CY"]=carry
            return False, "RRC"
        if op == 0x2F:
            self.reg["A"] = (~self.reg["A"]) & 0xFF
            return False, "CMA"

        # If unknown:
        return False, f"UNKNOWN OPCODE {op:02X}"

    def run(self, max_steps=100000):
        trace = []
        steps = 0
        while not self.halted and steps < max_steps:
            steps += 1
            halted, msg = self.step()
            trace.append((self.PC, msg))
            if halted: break
        return trace

#########################################
# assembler.py
# Very small assembler: labels, simple parsing, directives: ORG, DB, DW.
# Produces list of bytes. Supports a subset of mnemonics matching our emulator.

import re

# small opcode map for implemented instructions (subset)
OPC = {
    "NOP": [0x00],
    "HLT": [0x76],
    "RLC": [0x07], "RRC": [0x0F], "CMA": [0x2F],
    # MOV r1,r2  base 0x40 .. 0x7F
    # we'll compute MOV in code
    # MVI r, data (0x06 + r*0x08)
    # LXI
    "LXI B": [0x01], "LXI D": [0x11], "LXI H": [0x21], "LXI SP": [0x31],
    "LDA": [0x3A], "STA": [0x32], "LHLD": [0x2A], "SHLD": [0x22],
    "ADI": [0xC6], "SUI": [0xD6], "ANI": [0xE6], "ORI": [0xF6], "XRI": [0xEE],
    "JMP": [0xC3], "JNZ":[0xC2], "JZ":[0xCA], "JNC":[0xD2], "JC":[0xDA],
    "CALL":[0xCD], "RET":[0xC9],
    "PUSH B":[0xC5],"PUSH D":[0xD5],"PUSH H":[0xE5],"PUSH PSW":[0xF5],
    "POP B":[0xC1],"POP D":[0xD1],"POP H":[0xE1],"POP PSW":[0xF1],
    "IN": [0xDB], "OUT":[0xD3],
    "ADI": [0xC6], "CPI":[0xFE],
}

REG_ENC = {"B":0,"C":1,"D":2,"E":3,"H":4,"L":5,"M":6,"A":7}

def tokenize(line):
    # remove comments ; or //
    line = re.sub(r";.*","",line)
    line = re.sub(r"//.*","",line)
    return line.strip()

def assemble(text: str, base=0x0000):
    lines = text.splitlines()
    # First pass: labels
    addr = base
    labels = {}
    processed = []
    for raw in lines:
        s = tokenize(raw)
        if s=="":
            continue
        if ":" in s:
            lab, rest = s.split(":",1); labels[lab.strip()] = addr; s = rest.strip()
            if s=="":
                continue
        parts = s.split()
        mnemonic = parts[0].upper()
        # directives
        if mnemonic == "ORG":
            addr = int(parts[1],0); continue
        if mnemonic == "DB":
            # bytes
            args = " ".join(parts[1:]).split(",")
            for a in args:
                a=a.strip()
                if a.startswith("'") and a.endswith("'"):
                    b = ord(a[1:-1])
                else:
                    b = int(a,0)
                addr += 1
            processed.append(("DB", args))
            continue
        # estimate size for labels pass
        if mnemonic == "DB":
            pass
        elif mnemonic in ("LXI","LDA","STA","LHLD","SHLD","JMP","JNZ","JZ","JC","JNC","CALL"):
            if mnemonic in ("LXI",):
                addr += 3
            else:
                addr += 3
        elif mnemonic in ("MVI",):
            addr += 2
        elif mnemonic in ("ADI","SUI","ANI","ORI","XRI","CPI","IN","OUT"):
            addr += 2
        elif mnemonic in ("PUSH","POP","RET","RST","NOP","HLT","RLC","RRC","CMA"):
            addr += 1
        elif mnemonic == "MOV":
            addr += 1
        else:
            addr += 1
        processed.append((mnemonic, parts[1:] if len(parts)>1 else []))

    # Second pass: emit bytes
    addr = base
    out = []
    for item in processed:
        mnemonic = item[0]; args = item[1]
        if mnemonic == "DB":
            for a in args:
                a=a.strip()
                if a.startswith("'") and a.endswith("'"):
                    b = ord(a[1:-1])
                else:
                    b = int(a,0)
                out.append(b & 0xFF); addr+=1
            continue
        if mnemonic == "ORG":
            addr = int(args[0],0); continue

        if mnemonic == "MOV":
            # args like B,A
            a = "".join(args)
            dst,src = [x.strip().upper() for x in a.split(",")]
            opcode = 0x40 | (REG_ENC[dst]<<3) | REG_ENC[src]
            out.append(opcode); addr+=1; continue

        if mnemonic == "MVI":
            a = "".join(args)
            dst,imm = [x.strip().upper() for x in a.split(",")]
            opcode = 0x06 | (REG_ENC[dst]<<3)
            out.append(opcode); out.append(int(imm,0)&0xFF); addr+=2; continue

        if mnemonic in ("LXI",):
            rp = args[0].upper().strip(",")
            imm = int(args[1],0)
            if rp=="B": opc=0x01
            elif rp=="D": opc=0x11
            elif rp=="H": opc=0x21
            elif rp=="SP": opc=0x31
            else: opc=0x00
            out += [opc, imm & 0xFF, (imm>>8)&0xFF]; addr+=3; continue

        if mnemonic in ("LDA","STA","LHLD","SHLD","JMP","JNZ","JZ","JC","JNC","CALL"):
            label_or_addr = args[0]
            val = None
            if label_or_addr in labels:
                val = labels[label_or_addr]
            else:
                val = int(label_or_addr,0)
            baseopc = {"LDA":0x3A,"STA":0x32,"LHLD":0x2A,"SHLD":0x22,
                       "JMP":0xC3,"JNZ":0xC2,"JZ":0xCA,"JC":0xDA,"JNC":0xD2,"CALL":0xCD}[mnemonic]
            out += [baseopc, val & 0xFF, (val>>8)&0xFF]; addr+=3; continue

        if mnemonic in ("ADI","SUI","ANI","ORI","XRI","CPI","IN","OUT"):
            imm = args[0]
            if mnemonic in ("IN","OUT"):
                val = int(imm,0)
                opc = {"OUT":0xD3,"IN":0xDB}[mnemonic]
                out += [opc, val & 0xFF]; addr+=2; continue
            val = int(imm,0)
            opc = {"ADI":0xC6,"SUI":0xD6,"ANI":0xE6,"ORI":0xF6,"XRI":0xEE,"CPI":0xFE}[mnemonic]
            out += [opc, val & 0xFF]; addr+=2; continue

        if mnemonic in ("PUSH","POP"):
            arg = "".join(args).upper()
            m = {"PUSHB":0xC5,"PUSHD":0xD5,"PUSHH":0xE5,"PUSHP":0xF5,
                 "POPB":0xC1,"POPD":0xD1,"POPH":0xE1,"POPP":0xF1}
            key = mnemonic + arg
            # support PUSH B as "PUSH B"
            key = mnemonic + arg
            if key in {"PUSHB","PUSHD","PUSHH","PUSHP","POPB","POPD","POPH","POPP"}:
                opc = {"PUSHB":0xC5,"PUSHD":0xD5,"PUSHH":0xE5,"PUSHP":0xF5,
                       "POPB":0xC1,"POPD":0xD1,"POPH":0xE1,"POPP":0xF1}[key]
                out.append(opc); addr+=1; continue
            # alternate: PUSH PSW
            if arg=="PSW" and mnemonic=="PUSH":
                out.append(0xF5); addr+=1; continue
            if arg=="PSW" and mnemonic=="POP":
                out.append(0xF1); addr+=1; continue

        if mnemonic in ("RET","NOP","HLT","RLC","RRC","CMA"):
            out.append(OPC[mnemonic][0]); addr+=1; continue

        # default unknown: try opcode map
        if mnemonic in OPC:
            out += OPC[mnemonic]; addr += len(OPC[mnemonic]); continue

        # if unknown, ignore (could raise)
        raise ValueError(f"Unknown mnemonic '{mnemonic}'")

    return out

##########################################
@app.route("/8085", methods=["GET", "POST"])
def sim8085():
    result = None
    error = None

    # 8085 Register Set
    registers = {"A": 0, "B": 0, "C": 0, "D": 0, "E": 0, "H": 0, "L": 0}
    memory = [0] * 65536
    pc = 0  # Program Counter

    def execute_instruction(instr):
        parts = instr.strip().split()
        opcode = parts[0].upper()

        # MOV r1,r2
        if opcode == "MOV":
            dst, src = parts[1].split(",")
            registers[dst] = registers[src]

        # MVI r, data
        elif opcode == "MVI":
            dst, data = parts[1].split(",")
            registers[dst] = int(data, 16)

        # ADD r
        elif opcode == "ADD":
            r = parts[1]
            registers["A"] = (registers["A"] + registers[r]) & 0xFF

        # INR r
        elif opcode == "INR":
            r = parts[1]
            registers[r] = (registers[r] + 1) & 0xFF

        # HLT
        elif opcode == "HLT":
            return False

        else:
            raise Exception(f"Unsupported instruction: {instr}")

        return True

    if request.method == "POST":
        code = request.form.get("program")
        try:
            instructions = code.split("\n")

            for instr in instructions:
                instr = instr.strip()
                if instr == "":
                    continue
                if not execute_instruction(instr):
                    break

            result = {
                "registers": registers,
                "memory_sample": memory[:20]  # show small chunk
            }

        except Exception as e:
            error = str(e)

    return render_template("8085.html", result=result, error=error)

###################################################################################################
@app.route("/IDFT", methods=["GET", "POST"])
def IDFT():
    result = None

    if request.method == "POST":
        try:
            # --- Get input sequence from user ---
            seq_str = request.form["sequence"]
            seq_str = seq_str.replace(",", " ")
            X = np.array(list(map(complex, seq_str.split())))  # Allow complex input
            N = len(X)

            # --- Compute IDFT ---
            x_reconstructed = np.fft.ifft(X, N)

            # --- Prepare textual results ---
            idft_list = [
                f"n={n}: {x_reconstructed[n].real:.4f} + j{x_reconstructed[n].imag:.4f}"
                for n in range(N)
            ]

            # --- Prepare result dict for template ---
            result = {
                "sequence": seq_str,
                "idft": idft_list,
            }

        except Exception as e:
            result = {"error": str(e)}

    return render_template("IDFT.html", result=result)

####################################################################################################
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)



