"""Validate compiled paper artifacts without treating this as scientific approval."""
import argparse
import json
import re
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path


def output(*args):
    return subprocess.check_output(args, text=True)


def minimum_margins(width, height, boxes):
    """Return all four content margins, in PDF points (72 points = 1 inch)."""
    assert boxes, "No content bounding boxes"
    return dict(left=min(b[0] for b in boxes),
                bottom=min(b[1] for b in boxes),
                right=width-max(b[2] for b in boxes),
                top=height-max(b[3] for b in boxes))


def validate_margins(margins):
    assert min(margins.values()) >= 72, ("Content violates one-inch margins", margins)


def main_page_limit(stage):
    return 14 if stage == "revision" else 10


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--latex", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--without-response", action="store_true", help="Check a public bundle that deliberately excludes confidential reviews.")
    ap.add_argument("--submission-stage", choices=["initial", "revision", "resubmission"], default="initial",
                    help="Technical scenario, not verification of the editor's decision; RRQ/resubmission uses 10 pages.")
    args = ap.parse_args()
    report = {"scientific_submission_ready": None, "documents": {},
              "submission_stage": args.submission_stage,
              "scientific_assessment": "not performed by this technical check"}
    main_limit = main_page_limit(args.submission_stage)
    documents = [("main", main_limit), ("Supplementary_Material", 4)]
    if not args.without_response:
        documents.append(("Response_to_Reviewers", 20))
    for name, limit in documents:
        pdf = args.latex / (name + ".pdf")
        info = output("pdfinfo", str(pdf))
        pages = int(re.search(r"^Pages:\s+(\d+)", info, re.M)[1])
        assert pages <= limit, (name, pages, limit)
        log = (args.latex / (name + ".log")).read_text(errors="replace")
        issues = re.findall(r".*(?:undefined|Overfull|LaTeX Warning).*", log)
        assert not issues, (name, issues)
        bbox = output("pdftotext", "-bbox", str(pdf), "-")
        # Some math glyph encodings are emitted as XML-illegal control bytes;
        # replacing only their text preserves every word's bounding box.
        bbox = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "\ufffd", bbox)
        root = ET.fromstring(bbox)
        words = 0
        dimensions = []
        for page in root.iter("{http://www.w3.org/1999/xhtml}page"):
            w, h = float(page.attrib["width"]), float(page.attrib["height"])
            assert abs(w-612) < .01 and abs(h-792) < .01, ("Letter paper required", w, h)
            dimensions.append((w, h))
            for word in page.iter("{http://www.w3.org/1999/xhtml}word"):
                x0, y0, x1, y1 = (float(word.attrib[k]) for k in ("xMin", "yMin", "xMax", "yMax"))
                assert 0 <= x0 <= x1 <= w and 0 <= y0 <= y1 <= h, (name, word.text)
                words += 1
        # Ghostscript measures rendered ink, including equations, rules and graphics,
        # not just the text that pdftotext happens to decode.
        ink = subprocess.run(["gs", "-q", "-dSAFER", "-dNOPAUSE", "-dBATCH", "-sDEVICE=bbox", str(pdf)],
                             check=True, capture_output=True, text=True).stderr
        boxes = [tuple(map(float, row.split()))
                 for row in re.findall(r"^%%HiResBoundingBox: (.+)$", ink, re.M)]
        assert len(boxes) == pages == len(dimensions), (name, len(boxes), pages)
        per_page = [minimum_margins(w, h, [box]) for (w, h), box in zip(dimensions, boxes)]
        for page_number, margins in enumerate(per_page, 1):
            try:
                validate_margins(margins)
            except AssertionError as exc:
                raise AssertionError((name, page_number, str(exc))) from exc
        font_rows = output("pdffonts", str(pdf)).splitlines()[2:]
        assert font_rows and all(line.split()[-5] == "yes" for line in font_rows), (name, font_rows)
        assert all('Type 3' not in line for line in font_rows), (name, "Type 3 font")
        raster_rows = output("pdfimages", "-list", str(pdf)).splitlines()[2:]
        assert not raster_rows, (name, "Embedded raster images", raster_rows)
        report["documents"][name] = dict(pages=pages, words_within_page=words,
            page_limit=limit, minimum_ink_margin_points=min(min(m.values()) for m in per_page),
            per_page_ink_margins_points=per_page,
            no_undefined_overfull_or_latex_warnings=True, fonts_embedded=True,
            no_type3_fonts=True, no_embedded_raster_images=True)
    if not args.without_response:
        reply = (args.latex / "Response_to_Reviewers.tex").read_text()
        count = len(re.findall(r"^\\commentitem\{", reply, re.M))
        assert count == 20
        report["substantive_reply_items"] = count
        assert r"\section*{Appendix: Original Reviewer Reports}" in reply
        report["original_review_appendix_present"] = True
    manuscript = (args.latex / "bare_jrnl.tex").read_text()
    abstract = manuscript.split(r"\begin{abstract}")[1].split(r"\end{abstract}")[0]
    abstract_words = len(abstract.split())
    assert 150 <= abstract_words <= 250, abstract_words
    report["abstract_source_word_count"] = abstract_words
    main_aux = (args.latex / "main.aux").read_text()
    supp_aux = (args.latex / "Supplementary_Material.aux").read_text()
    def table_numbers(aux):
        values = {'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100, 'D': 500, 'M': 1000}
        numbers = set()
        for roman in re.findall(r"\\newlabel\{tab:[^}]+\}\{\{([IVXLCDM]+)\}", aux):
            total = previous = 0
            for c in reversed(roman):
                current = values[c]
                total += -current if current < previous else current
                previous = max(previous, current)
            numbers.add(total)
        return numbers
    main_tables, supp_tables = table_numbers(main_aux), table_numbers(supp_aux)
    assert main_tables and supp_tables
    assert main_tables == set(range(1, max(main_tables)+1)), main_tables
    assert supp_tables == set(range(max(main_tables)+1, max(supp_tables)+1)), supp_tables
    report['table_number_ranges'] = dict(main=[min(main_tables), max(main_tables)],
                                       supplement=[min(supp_tables), max(supp_tables)])
    last_main = int(re.search(r'\\newlabel\{eq:kd_gain\}\{\{(\d+)\}', main_aux)[1])
    first_supp = int(re.search(r'\\newlabel\{eq:health_se\}\{\{(\d+)\}', supp_aux)[1])
    assert first_supp == last_main + 1, (last_main, first_supp)
    report['equation_transition'] = dict(last_main=last_main, first_supplement=first_supp)
    historical_figures = ['fig:pipeline', 'fig:cv_health_vs_gain',
        'fig:cv_perclass_kd_diagnostic', 'fig:cv_privacy_sweep_kd',
        'fig:cv_weak_teacher_tcrd', 'fig:vctk_iemocap_summary',
        'fig:app_vctk_main', 'fig:app_iemocap_main']
    historical_tables = ['tab:cv_accent_baseline_dp_loss', 'tab:cv_release_routing',
        'tab:cross_dataset_summary', 'tab:app_data_stats', 'tab:app_default_hyperparams',
        'tab:app_cv_routing_diagnostics', 'tab:app_cv_privacy_sweep_gain']
    for label in historical_figures + historical_tables:
        assert f'\\newlabel{{{label}}}' in main_aux, ('Historical view absent from main', label)
    report['historical_views_in_main'] = dict(figures=len(historical_figures), tables=len(historical_tables))
    report["consecutive_main_supplement_numbering"] = True
    args.out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
