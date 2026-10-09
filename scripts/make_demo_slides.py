"""Generate 4 dark 'lecture slides' with real formulas, for a YouTube-free demo of the screenshots mode.

Usage (matplotlib needed: pip install matplotlib):
    python scripts/make_demo_slides.py demo_slides/
    python -m lecturify images demo_slides/ --title "Calculus I — derivatives" --context-file demo_slides/context.txt --verify
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

SLIDES = [
    ("The derivative", [r"$f'(x) = \lim_{h \to 0} \frac{f(x+h) - f(x)}{h}$"], "tangent"),
    ("Power rule", [r"$\frac{d}{dx} x^n = n\,x^{n-1}$", r"$\frac{d}{dx} x^3 = 3x^2$"], None),
    ("Chain rule", [r"$\frac{d}{dx} g(h(x)) = g'(h(x))\,h'(x)$", r"$\frac{d}{dx}\sin(x^2) = 2x\cos(x^2)$"], None),
    ("Gradient", [r"$\nabla f = \left(\frac{\partial f}{\partial x}, \frac{\partial f}{\partial y}\right)$"], "field"),
]
CONTEXT = ("Lecture 4 of Calculus I. Today: the derivative as a limit, then the power rule, the chain rule, "
           "and a first look at gradients for functions of two variables.\n")


def main(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for i, (title, formulas, kind) in enumerate(SLIDES, 1):
        fig = plt.figure(figsize=(10.24, 5.76), dpi=100, facecolor="#14141c")
        fig.text(0.05, 0.88, title, color="white", fontsize=30)
        for j, f in enumerate(formulas):
            fig.text(0.05, 0.65 - 0.18 * j, f, color="#9cd0ff", fontsize=26)
        if kind:
            ax = fig.add_axes([0.6, 0.08, 0.36, 0.5], facecolor="#14141c")
            if kind == "tangent":
                x = np.linspace(-1, 2, 100)
                ax.plot(x, x ** 2, c="#ffd166", lw=3)
                ax.plot(x, 2 * x - 1, c="#ef476f", lw=2)
            else:
                X, Y = np.meshgrid(np.linspace(-2, 2, 9), np.linspace(-2, 2, 9))
                ax.quiver(X, Y, 2 * X, 2 * Y, color="#06d6a0")
            ax.axis("off")
        fig.savefig(out / f"slide_{i}.png", facecolor=fig.get_facecolor())
        plt.close(fig)
    (out / "context.txt").write_text(CONTEXT)
    print(f"Wrote {len(SLIDES)} slides + context.txt to {out}/")


if __name__ == "__main__":
    main(Path(sys.argv[1] if len(sys.argv) > 1 else "demo_slides"))
