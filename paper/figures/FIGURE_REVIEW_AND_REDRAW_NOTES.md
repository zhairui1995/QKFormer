# Figure Review And Redraw Notes

Status date: 2026-06-11

## Gemini Availability

The local environment has no callable Gemini CLI (`gemini` is not on `PATH`),
and the available Codex plugins do not include a Gemini connector. Chrome may
have Gemini available to the user, but this Codex session does not currently
have a reliable Chrome-control or Gemini API path that can return a model
review. Therefore, no claim is made that Gemini reviewed these figures.

## Internal Review Verdict

The first generated figure set was technically correct and vector-only, but
Figures 1 and 2 looked like schematic drafts rather than polished paper
figures. Figures 3 and 4 were scientifically appropriate but needed stronger
publication styling and clearer dataset/claim separation.

## Redraw Actions

- Figure 1 was redrawn as a structured audit-ladder graphic with separated
  evidence path, negative-control row, and claim-boundary row.
- Figure 2 was redrawn to make correct-address, shuffled-address, and
  coarse/global controls visually comparable and less decorative.
- Figure 3 was refined with a dataset separator and explicit positive vs
  shuffled-control panels.
- Figure 4 was refined with a shaded address-over-token gap and an annotation
  showing the limited seed-42 calibration-efficiency evidence.

## Remaining Design Boundary

The figures are intentionally restrained: white background, thin lines,
color-blind-friendly palette, no gradients, no 3D, no hardware or speed visual
metaphors. This matches the current `GO-AUDIT` paper identity and avoids
implying residual-adapter or hardware-acceleration claims.

## When Visio/Figma Would Help

Manual design software would only be worth using if the paper needs a highly
custom visual metaphor or a publisher-style graphical abstract. It is not
required for the current submission figures. A script-based vector workflow is
preferable for traceability and fast regeneration after experiment updates.
