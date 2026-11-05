# AcademicForge design

AcademicForge is a working tool. People come to it with a research question, scan a list of papers, pick a few, and read a plan. The interface should get out of the way of that task, so the look is restrained: one dark surface, one accent, one type family.

The source of truth is `.streamlit/config.toml` (Streamlit theme) and `frontend/styles.css` (tokens and components). Change them together.

## Color

| Token | Value | Use |
| :--- | :--- | :--- |
| `--af-bg` | `#0E1117` | Page background. Streamlit's dark base, pinned so it does not follow the viewer's OS theme. |
| `--af-panel` | `#161A23` | Cards, inputs, the plan panel. |
| `--af-panel-hover` | `#1B2030` | Card hover. |
| `--af-line` | `rgba(255,255,255,0.08)` | Borders and dividers. |
| `--af-line-strong` | `rgba(255,255,255,0.16)` | Focused or selected borders. |
| `--af-ember` | `#ED1C24` | The accent. Primary buttons, the selected state, focus rings, text selection. Nothing decorative. |
| `--af-ember-hover` | `#C4151B` | Primary button hover. |
| `--af-ember-soft` | `rgba(237,28,36,0.14)` | Selected card tint and selection highlight. |
| `--af-text` | `#F0F0EE` | Body and headings. |
| `--af-text-mid` | `rgba(240,240,238,0.72)` | Metadata and secondary text (passes 4.5:1 on `--af-bg`). |
| `--af-text-low` | `rgba(240,240,238,0.56)` | Hints only, never body copy. |

Paper categories (Foundational, Survey, and so on) are shown as plain muted text. They do not get their own colors, because color is reserved for state.

## Type

One family: Inter, falling back to the system sans. Fixed rem sizes on a 1.2 scale, no fluid type.

| Role | Size | Weight |
| :--- | :--- | :--- |
| Wordmark | 1.25rem | 650 |
| Section heading | 1.1rem | 600 |
| Paper title | 1rem | 600 |
| Body | 0.95rem | 400 |
| Metadata, captions | 0.82rem | 400 |

Headings use sentence case. Prose runs at most 72ch wide.

## Layout

A single centered column, 880px wide, in the order of the task:

1. Wordmark and one line saying what the tool does.
2. The question box and Search button on one row, with field, lens and mode as three compact selects below it.
3. Papers: a list of cards. Each card shows the title, one metadata line, the category, and a collapsed abstract, with "Use in plan", Summary and Guidance on one row.
4. Research plan: a bar with the selection count and the Write plan button, then the plan itself.

No centered hero, no eyebrow labels, no nested cards, no colored side borders.

## Motion

Motion only shows state. Transitions run 160ms with `cubic-bezier(0.2, 0, 0, 1)`.

- Buttons darken on hover and press down 1px on click.
- Cards brighten their border on hover. A selected card keeps an ember border.
- Loading uses a three-dot pulse inline where the result will appear.
- Nothing animates on scroll or on page load.
- Under `prefers-reduced-motion`, all transitions and the pulse stop.

## Browser surfaces

Text selection, focus rings, scrollbars and the caret use the palette: ember selection on `--af-ember-soft`, a 2px ember focus ring, thin dark scrollbars.

## Copy

Plain and specific. Buttons name their action ("Search", "Write plan"). Errors say what went wrong and what to do next. No em dashes, no hype words, no "not X but Y". The LLM prompts carry the same rule so generated summaries and plans match the interface.
