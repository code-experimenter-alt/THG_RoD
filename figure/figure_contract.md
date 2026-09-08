# Figure 1: original layout, vector delivery

The source layout is `figure/new_fra_fig.png` in `TMM_Yadi_oldversion.zip`.
The existing panels, arrows, colors and typography are retained as outlined
vector contours in `fig1_old_base.svg`; this is not a newly designed diagram.
`refine_old_pipeline.py` applies only local corrections: the privacy note,
optional offline query, independent hard-label source, low-health wording,
and the inclusive high threshold.

Final assets are `fig1_old_vector.svg` and `fig1_old_vector.pdf`. Neither
contains a bitmap. Original lettering is outlined; correction labels remain
SVG text. The temporary generated edit is not used in either final asset.
The canvas is 6.455 by 2.98866 inches, placed at 84% of main-text width.

To regenerate the PDF/SVG from the included vector base, install CairoSVG
and run `python figure/refine_old_pipeline.py`. Paper compilation only
needs the supplied PDF. No PNG, generated-image service or tracing library is
required to compile or reapply these corrections.

The main PDF check verifies no embedded raster images, no Type 3 fonts,
embedded fonts, and one-inch content margins. The original-layout figure and
the compiled page are also inspected visually.
