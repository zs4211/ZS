from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK, WD_LINE_SPACING
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "artifacts" / "tensor_research_plan"
FIG_DIR = OUT_DIR / "figures"
DOCX_PATH = OUT_DIR / "交叉波束几何感知的物理约束低秩张量风场反演研究方案.docx"


# narrative_proposal preset, with a named East-Asian-font override for Chinese.
LATIN_FONT = "Calibri"
EAST_ASIA_FONT = "Microsoft YaHei"
MATH_FONT = "Cambria Math"
BLUE = "2E74B5"
DARK_BLUE = "1F4D78"
NAVY = "20374C"
MUTED = "667085"
LIGHT_FILL = "F4F6F9"
TABLE_HEADER = "EAF0F6"
GOLD = "B38322"
RED = "9B1C1C"
GREEN = "256D4A"
BLACK = "202124"
USABLE_DXA = 9360
TABLE_INDENT_DXA = 120


def set_cell_shading(cell, fill: str):
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = tc_pr.find(qn("w:shd"))
    if shd is None:
        shd = OxmlElement("w:shd")
        tc_pr.append(shd)
    shd.set(qn("w:fill"), fill)


def set_cell_margins(cell, top=80, start=120, bottom=80, end=120):
    tc = cell._tc
    tc_pr = tc.get_or_add_tcPr()
    tc_mar = tc_pr.first_child_found_in("w:tcMar")
    if tc_mar is None:
        tc_mar = OxmlElement("w:tcMar")
        tc_pr.append(tc_mar)
    for m, value in (("top", top), ("start", start), ("bottom", bottom), ("end", end)):
        node = tc_mar.find(qn(f"w:{m}"))
        if node is None:
            node = OxmlElement(f"w:{m}")
            tc_mar.append(node)
        node.set(qn("w:w"), str(value))
        node.set(qn("w:type"), "dxa")


def set_table_geometry(table, widths_dxa: list[int]):
    if sum(widths_dxa) != USABLE_DXA:
        raise ValueError(f"table widths must sum to {USABLE_DXA}: {widths_dxa}")
    table.autofit = False
    tbl_pr = table._tbl.tblPr

    tbl_w = tbl_pr.find(qn("w:tblW"))
    if tbl_w is None:
        tbl_w = OxmlElement("w:tblW")
        tbl_pr.append(tbl_w)
    tbl_w.set(qn("w:w"), str(USABLE_DXA))
    tbl_w.set(qn("w:type"), "dxa")

    tbl_ind = tbl_pr.find(qn("w:tblInd"))
    if tbl_ind is None:
        tbl_ind = OxmlElement("w:tblInd")
        tbl_pr.append(tbl_ind)
    tbl_ind.set(qn("w:w"), str(TABLE_INDENT_DXA))
    tbl_ind.set(qn("w:type"), "dxa")

    layout = tbl_pr.find(qn("w:tblLayout"))
    if layout is None:
        layout = OxmlElement("w:tblLayout")
        tbl_pr.append(layout)
    layout.set(qn("w:type"), "fixed")

    grid = table._tbl.tblGrid
    for child in list(grid):
        grid.remove(child)
    for width in widths_dxa:
        col = OxmlElement("w:gridCol")
        col.set(qn("w:w"), str(width))
        grid.append(col)

    for row in table.rows:
        for i, cell in enumerate(row.cells):
            cell.width = Inches(widths_dxa[i] / 1440)
            tc_pr = cell._tc.get_or_add_tcPr()
            tc_w = tc_pr.find(qn("w:tcW"))
            if tc_w is None:
                tc_w = OxmlElement("w:tcW")
                tc_pr.append(tc_w)
            tc_w.set(qn("w:w"), str(widths_dxa[i]))
            tc_w.set(qn("w:type"), "dxa")
            set_cell_margins(cell)
            cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER


def set_run_font(run, size=None, bold=None, italic=None, color=None, east_asia=None, latin=None):
    latin = latin or LATIN_FONT
    east_asia = east_asia or EAST_ASIA_FONT
    run.font.name = latin
    r_pr = run._element.get_or_add_rPr()
    fonts = r_pr.rFonts
    if fonts is None:
        fonts = OxmlElement("w:rFonts")
        r_pr.insert(0, fonts)
    fonts.set(qn("w:ascii"), latin)
    fonts.set(qn("w:hAnsi"), latin)
    fonts.set(qn("w:eastAsia"), east_asia)
    if size is not None:
        run.font.size = Pt(size)
    if bold is not None:
        run.bold = bold
    if italic is not None:
        run.italic = italic
    if color is not None:
        run.font.color.rgb = RGBColor.from_string(color)


def add_field(paragraph, instruction: str, placeholder=""):
    run = paragraph.add_run()
    begin = OxmlElement("w:fldChar")
    begin.set(qn("w:fldCharType"), "begin")
    instr = OxmlElement("w:instrText")
    instr.set(qn("xml:space"), "preserve")
    instr.text = instruction
    sep = OxmlElement("w:fldChar")
    sep.set(qn("w:fldCharType"), "separate")
    text = OxmlElement("w:t")
    text.text = placeholder
    end = OxmlElement("w:fldChar")
    end.set(qn("w:fldCharType"), "end")
    run._r.extend([begin, instr, sep, text, end])


def add_hyperlink(paragraph, text: str, url: str):
    part = paragraph.part
    rel_id = part.relate_to(url, "http://schemas.openxmlformats.org/officeDocument/2006/relationships/hyperlink", is_external=True)
    hyperlink = OxmlElement("w:hyperlink")
    hyperlink.set(qn("r:id"), rel_id)
    run = OxmlElement("w:r")
    r_pr = OxmlElement("w:rPr")
    color = OxmlElement("w:color")
    color.set(qn("w:val"), BLUE)
    underline = OxmlElement("w:u")
    underline.set(qn("w:val"), "single")
    fonts = OxmlElement("w:rFonts")
    fonts.set(qn("w:ascii"), LATIN_FONT)
    fonts.set(qn("w:hAnsi"), LATIN_FONT)
    fonts.set(qn("w:eastAsia"), EAST_ASIA_FONT)
    r_pr.extend([fonts, color, underline])
    run.append(r_pr)
    t = OxmlElement("w:t")
    t.text = text
    run.append(t)
    hyperlink.append(run)
    paragraph._p.append(hyperlink)


def add_numbering_definitions(doc: Document):
    numbering = doc.part.numbering_part.element
    existing_abs = [int(x.get(qn("w:abstractNumId"))) for x in numbering.findall(qn("w:abstractNum"))]
    existing_num = [int(x.get(qn("w:numId"))) for x in numbering.findall(qn("w:num"))]
    abs_start = max(existing_abs + [20]) + 1
    num_start = max(existing_num + [20]) + 1

    def make_abstract(abs_id: int, fmt: str, text: str, left: int, hanging: int):
        abstract = OxmlElement("w:abstractNum")
        abstract.set(qn("w:abstractNumId"), str(abs_id))
        multi = OxmlElement("w:multiLevelType")
        multi.set(qn("w:val"), "singleLevel")
        abstract.append(multi)
        lvl = OxmlElement("w:lvl")
        lvl.set(qn("w:ilvl"), "0")
        start = OxmlElement("w:start")
        start.set(qn("w:val"), "1")
        num_fmt = OxmlElement("w:numFmt")
        num_fmt.set(qn("w:val"), fmt)
        lvl_text = OxmlElement("w:lvlText")
        lvl_text.set(qn("w:val"), text)
        suff = OxmlElement("w:suff")
        suff.set(qn("w:val"), "tab")
        p_pr = OxmlElement("w:pPr")
        tabs = OxmlElement("w:tabs")
        tab = OxmlElement("w:tab")
        tab.set(qn("w:val"), "num")
        tab.set(qn("w:pos"), str(left))
        tabs.append(tab)
        ind = OxmlElement("w:ind")
        ind.set(qn("w:left"), str(left))
        ind.set(qn("w:hanging"), str(hanging))
        spacing = OxmlElement("w:spacing")
        spacing.set(qn("w:after"), "80")
        spacing.set(qn("w:line"), "290")
        spacing.set(qn("w:lineRule"), "auto")
        p_pr.extend([tabs, ind, spacing])
        lvl.extend([start, num_fmt, lvl_text, suff, p_pr])
        abstract.append(lvl)
        numbering.append(abstract)

    def make_num(num_id: int, abs_id: int):
        num = OxmlElement("w:num")
        num.set(qn("w:numId"), str(num_id))
        ref = OxmlElement("w:abstractNumId")
        ref.set(qn("w:val"), str(abs_id))
        num.append(ref)
        numbering.append(num)

    make_abstract(abs_start, "bullet", "•", 540, 280)
    make_abstract(abs_start + 1, "decimal", "%1.", 540, 280)
    make_num(num_start, abs_start)
    make_num(num_start + 1, abs_start + 1)
    return num_start, num_start + 1


def apply_num(paragraph, num_id: int):
    p_pr = paragraph._p.get_or_add_pPr()
    num_pr = p_pr.find(qn("w:numPr"))
    if num_pr is None:
        num_pr = OxmlElement("w:numPr")
        p_pr.append(num_pr)
    ilvl = OxmlElement("w:ilvl")
    ilvl.set(qn("w:val"), "0")
    nid = OxmlElement("w:numId")
    nid.set(qn("w:val"), str(num_id))
    num_pr.extend([ilvl, nid])


def restart_numbering(doc, template_num_id: int) -> int:
    """Create a fresh numbering instance that starts again at 1."""
    numbering = doc.part.numbering_part.element
    template = next(
        n for n in numbering.findall(qn("w:num"))
        if int(n.get(qn("w:numId"))) == template_num_id
    )
    abstract_id = template.find(qn("w:abstractNumId")).get(qn("w:val"))
    existing = [int(n.get(qn("w:numId"))) for n in numbering.findall(qn("w:num"))]
    new_id = max(existing) + 1
    num = OxmlElement("w:num")
    num.set(qn("w:numId"), str(new_id))
    ref = OxmlElement("w:abstractNumId")
    ref.set(qn("w:val"), abstract_id)
    num.append(ref)
    level_override = OxmlElement("w:lvlOverride")
    level_override.set(qn("w:ilvl"), "0")
    start_override = OxmlElement("w:startOverride")
    start_override.set(qn("w:val"), "1")
    level_override.append(start_override)
    num.append(level_override)
    numbering.append(num)
    return new_id


def add_bullet(doc, text, bullet_id, bold_lead=None):
    p = doc.add_paragraph()
    apply_num(p, bullet_id)
    p.paragraph_format.space_after = Pt(4)
    p.paragraph_format.line_spacing = 1.208
    if bold_lead and text.startswith(bold_lead):
        r = p.add_run(bold_lead)
        set_run_font(r, bold=True)
        r = p.add_run(text[len(bold_lead):])
        set_run_font(r)
    else:
        r = p.add_run(text)
        set_run_font(r)
    return p


def add_number(doc, text, number_id, bold_lead=None):
    p = doc.add_paragraph()
    apply_num(p, number_id)
    p.paragraph_format.space_after = Pt(4)
    p.paragraph_format.line_spacing = 1.208
    if bold_lead and text.startswith(bold_lead):
        r = p.add_run(bold_lead)
        set_run_font(r, bold=True)
        r = p.add_run(text[len(bold_lead):])
        set_run_font(r)
    else:
        r = p.add_run(text)
        set_run_font(r)
    return p


def add_para(doc, text="", *, bold_lead=None, italic=False, align=None, color=BLACK, keep_with_next=False):
    p = doc.add_paragraph()
    if align is not None:
        p.alignment = align
    p.paragraph_format.keep_with_next = keep_with_next
    if bold_lead and text.startswith(bold_lead):
        r = p.add_run(bold_lead)
        set_run_font(r, bold=True, color=color)
        r = p.add_run(text[len(bold_lead):])
        set_run_font(r, italic=italic, color=color)
    else:
        r = p.add_run(text)
        set_run_font(r, italic=italic, color=color)
    return p


def add_equation(doc, text: str, number: str | None = None):
    table = doc.add_table(rows=1, cols=2)
    set_table_geometry(table, [8500, 860])
    table.style = "Table Grid"
    for cell in table.rows[0].cells:
        tc_pr = cell._tc.get_or_add_tcPr()
        borders = tc_pr.find(qn("w:tcBorders"))
        if borders is None:
            borders = OxmlElement("w:tcBorders")
            tc_pr.append(borders)
        for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
            tag = borders.find(qn(f"w:{edge}"))
            if tag is None:
                tag = OxmlElement(f"w:{edge}")
                borders.append(tag)
            tag.set(qn("w:val"), "nil")
    p = table.cell(0, 0).paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(2)
    run = p.add_run(text)
    set_run_font(run, size=10.5, latin=MATH_FONT, east_asia=EAST_ASIA_FONT)
    p2 = table.cell(0, 1).paragraphs[0]
    p2.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    run = p2.add_run(number or "")
    set_run_font(run, size=10, latin=MATH_FONT)
    return table


def add_callout(doc, label: str, text: str, color=BLUE):
    table = doc.add_table(rows=1, cols=1)
    tr_pr = table.rows[0]._tr.get_or_add_trPr()
    tr_pr.append(OxmlElement("w:cantSplit"))
    set_table_geometry(table, [USABLE_DXA])
    set_cell_shading(table.cell(0, 0), LIGHT_FILL)
    cell = table.cell(0, 0)
    p = cell.paragraphs[0]
    p.paragraph_format.keep_together = True
    p.paragraph_format.space_after = Pt(3)
    r = p.add_run(label + "  ")
    set_run_font(r, bold=True, color=color)
    r = p.add_run(text)
    set_run_font(r, color=BLACK)
    # Restrained left border only.
    tc_pr = cell._tc.get_or_add_tcPr()
    borders = OxmlElement("w:tcBorders")
    left = OxmlElement("w:left")
    left.set(qn("w:val"), "single")
    left.set(qn("w:sz"), "18")
    left.set(qn("w:color"), color)
    borders.append(left)
    tc_pr.append(borders)
    doc.add_paragraph().paragraph_format.space_after = Pt(0)


def add_table(doc, headers, rows, widths, font_size=9.2):
    table = doc.add_table(rows=1, cols=len(headers))
    table.style = "Table Grid"
    table.rows[0]._tr.get_or_add_trPr().append(OxmlElement("w:tblHeader"))
    for i, header in enumerate(headers):
        cell = table.rows[0].cells[i]
        set_cell_shading(cell, TABLE_HEADER)
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_after = Pt(0)
        r = p.add_run(str(header))
        set_run_font(r, size=font_size, bold=True, color=NAVY)
    for row in rows:
        cells = table.add_row().cells
        for i, value in enumerate(row):
            p = cells[i].paragraphs[0]
            p.paragraph_format.space_after = Pt(0)
            p.paragraph_format.line_spacing = 1.12
            if i == 0 and len(headers) > 2:
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            r = p.add_run(str(value))
            set_run_font(r, size=font_size)
    set_table_geometry(table, widths)
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.space_after = Pt(4)
    return table


def set_image_alt(inline_shape, title: str, description: str):
    doc_pr = inline_shape._inline.docPr
    doc_pr.set("title", title)
    doc_pr.set("descr", description)


def add_figure(doc, path: Path, caption: str, width=6.2, alt=""):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.keep_with_next = True
    shape = p.add_run().add_picture(str(path), width=Inches(width))
    set_image_alt(shape, caption, alt or caption)
    cap = doc.add_paragraph(style="Caption")
    cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    cap.paragraph_format.space_before = Pt(3)
    cap.paragraph_format.space_after = Pt(8)
    r = cap.add_run(caption)
    set_run_font(r, size=9, color=MUTED)


def setup_styles(doc: Document):
    section = doc.sections[0]
    section.page_width = Inches(8.5)
    section.page_height = Inches(11)
    section.top_margin = Inches(1)
    section.bottom_margin = Inches(1)
    section.left_margin = Inches(1)
    section.right_margin = Inches(1)
    section.header_distance = Inches(0.492)
    section.footer_distance = Inches(0.492)

    normal = doc.styles["Normal"]
    normal.font.name = LATIN_FONT
    normal._element.rPr.rFonts.set(qn("w:ascii"), LATIN_FONT)
    normal._element.rPr.rFonts.set(qn("w:hAnsi"), LATIN_FONT)
    normal._element.rPr.rFonts.set(qn("w:eastAsia"), EAST_ASIA_FONT)
    normal.font.size = Pt(11)
    normal.font.color.rgb = RGBColor.from_string(BLACK)
    pf = normal.paragraph_format
    pf.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    pf.space_before = Pt(0)
    pf.space_after = Pt(8)
    pf.line_spacing = 1.333
    pf.widow_control = True

    for name, size, color, before, after in (
        ("Heading 1", 16, BLUE, 18, 10),
        ("Heading 2", 13, BLUE, 12, 6),
        ("Heading 3", 12, DARK_BLUE, 8, 4),
    ):
        style = doc.styles[name]
        style.font.name = LATIN_FONT
        style._element.rPr.rFonts.set(qn("w:ascii"), LATIN_FONT)
        style._element.rPr.rFonts.set(qn("w:hAnsi"), LATIN_FONT)
        style._element.rPr.rFonts.set(qn("w:eastAsia"), EAST_ASIA_FONT)
        style.font.size = Pt(size)
        style.font.bold = True
        style.font.color.rgb = RGBColor.from_string(color)
        style.paragraph_format.space_before = Pt(before)
        style.paragraph_format.space_after = Pt(after)
        style.paragraph_format.keep_with_next = True
        style.paragraph_format.keep_together = True

    title = doc.styles["Title"]
    title.font.name = LATIN_FONT
    title._element.rPr.rFonts.set(qn("w:eastAsia"), EAST_ASIA_FONT)
    title.font.size = Pt(28)
    title.font.bold = True
    title.font.color.rgb = RGBColor.from_string(NAVY)

    subtitle = doc.styles["Subtitle"]
    subtitle.font.name = LATIN_FONT
    subtitle._element.rPr.rFonts.set(qn("w:eastAsia"), EAST_ASIA_FONT)
    subtitle.font.size = Pt(14)
    subtitle.font.color.rgb = RGBColor.from_string(DARK_BLUE)

    caption = doc.styles["Caption"]
    caption.font.name = LATIN_FONT
    caption._element.rPr.rFonts.set(qn("w:eastAsia"), EAST_ASIA_FONT)
    caption.font.size = Pt(9)
    caption.font.color.rgb = RGBColor.from_string(MUTED)
    # The caption stays with the figure through the figure paragraph's
    # keep-with-next setting; it must not pull the next heading forward.
    caption.paragraph_format.keep_with_next = False

    # Request field updates when opened in Word/LibreOffice.
    settings = doc.settings._element
    update = settings.find(qn("w:updateFields"))
    if update is None:
        update = OxmlElement("w:updateFields")
        settings.append(update)
    update.set(qn("w:val"), "true")


def setup_header_footer(doc: Document):
    section = doc.sections[0]
    header = section.header
    p = header.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    p.paragraph_format.space_after = Pt(0)
    r = p.add_run("研究方案  |  物理约束低秩张量多普勒风场反演")
    set_run_font(r, size=8.5, color=MUTED)

    footer = section.footer
    p = footer.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    p.paragraph_format.space_after = Pt(0)
    r = p.add_run("第 ")
    set_run_font(r, size=8.5, color=MUTED)
    add_field(p, " PAGE ", "1")
    r = p.add_run(" 页 / 共 ")
    set_run_font(r, size=8.5, color=MUTED)
    add_field(p, " NUMPAGES ", "1")
    r = p.add_run(" 页")
    set_run_font(r, size=8.5, color=MUTED)


def make_figures():
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    font_path = r"C:\Windows\Fonts\msyh.ttc"
    title_font = ImageFont.truetype(font_path, 42)
    h_font = ImageFont.truetype(font_path, 31)
    body_font = ImageFont.truetype(font_path, 22)
    small_font = ImageFont.truetype(font_path, 18)

    def centered(draw, box, text, font, fill):
        left, top, right, bottom = box
        bbox = draw.multiline_textbbox((0, 0), text, font=font, spacing=4, align="center")
        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        draw.multiline_text(((left + right - w) / 2, (top + bottom - h) / 2), text,
                            font=font, fill=fill, spacing=4, align="center")

    def arrow(draw, start, end, color="#2E74B5", width=5, dashed=False):
        x1, y1 = start
        x2, y2 = end
        if dashed:
            n = 16
            for i in range(0, n, 2):
                xa = x1 + (x2 - x1) * i / n
                ya = y1 + (y2 - y1) * i / n
                xb = x1 + (x2 - x1) * (i + 1) / n
                yb = y1 + (y2 - y1) * (i + 1) / n
                draw.line((xa, ya, xb, yb), fill=color, width=width)
        else:
            draw.line((x1, y1, x2, y2), fill=color, width=width)
        ang = math.atan2(y2 - y1, x2 - x1)
        ah = 17
        pts = [(x2, y2),
               (x2 - ah * math.cos(ang - 0.5), y2 - ah * math.sin(ang - 0.5)),
               (x2 - ah * math.cos(ang + 0.5), y2 - ah * math.sin(ang + 0.5))]
        draw.polygon(pts, fill=color)

    # Figure 1: end-to-end scientific workflow.
    img = Image.new("RGB", (1980, 936), "white")
    d = ImageDraw.Draw(img)
    centered(d, (0, 20, 1980, 100), "从气象数据流到可验证张量反演的双轨架构", title_font, "#20374C")
    top_blocks = [
        (50, 245, 335, 420, "雷达观测\nVr / Z / 几何 / 时间"),
        (395, 245, 680, 420, "质量控制\n退模糊 / 掩膜 / 误差"),
        (740, 245, 1040, 420, "观测算子\nH 与 H*（矩阵无关）"),
        (1100, 245, 1400, 420, "张量反演\n低秩 + 连续方程"),
        (1460, 245, 1925, 420, "风场与置信产品\nu,v,w / 条件数 / 不确定度"),
    ]
    for b in top_blocks:
        box = b[:4]
        d.rounded_rectangle(box, radius=18, fill="#F4F6F9", outline="#2E74B5", width=4)
        centered(d, box, b[4], h_font if "\n" not in b[4] else body_font, "#20374C")
    for a, b in zip(top_blocks[:-1], top_blocks[1:]):
        arrow(d, (a[2] + 8, (a[1] + a[3]) // 2), (b[0] - 8, (b[1] + b[3]) // 2))
    bottom_blocks = [
        (275, 650, 660, 820, "PyDDA 基线\n现有 3DVAR 与残差"),
        (800, 650, 1185, 820, "OSSE 真值实验\n涡旋族 / 噪声 / 缺测"),
        (1325, 650, 1710, 820, "真实过程验证\n留出观测 / M / 人工判识"),
    ]
    for b in bottom_blocks:
        d.rounded_rectangle(b[:4], radius=18, fill="#FFF9ED", outline="#B38322", width=4)
        centered(d, b[:4], b[4], body_font, "#20374C")
    for b, target_x in zip(bottom_blocks, (1240, 1300, 1550)):
        arrow(d, ((b[0] + b[2]) // 2, b[1] - 6), (target_x, 435), color="#B38322", width=4, dashed=True)
    img.save(FIG_DIR / "workflow.png", dpi=(180, 180))

    # Figure 2: beam geometry conditioning on a logarithmic y-axis.
    img = Image.new("RGB", (1728, 936), "white")
    d = ImageDraw.Draw(img)
    centered(d, (0, 15, 1728, 90), "双雷达局地水平风反演的几何病态性", title_font, "#20374C")
    left, top, right, bottom = 165, 150, 1630, 780
    d.rectangle((left, top, right, bottom), outline="#667085", width=3)
    xmap = lambda th: left + (right - left) * th / 180.0
    ymin, ymax = math.log10(0.8), math.log10(20.0)
    ymap = lambda val: bottom - (bottom - top) * (math.log10(val) - ymin) / (ymax - ymin)
    d.rectangle((xmap(0), top, xmap(20), bottom), fill="#F9E5E5")
    d.rectangle((xmap(160), top, xmap(180), bottom), fill="#F9E5E5")
    d.rectangle((xmap(30), top, xmap(150), bottom), fill="#ECF6F0")
    for yv in (1, 2, 5, 10, 20):
        y = ymap(yv)
        d.line((left, y, right, y), fill="#D6DBE1", width=2)
        d.text((90, y - 12), str(yv), font=small_font, fill="#667085")
    for xv in range(0, 181, 30):
        x = xmap(xv)
        d.line((x, top, x, bottom), fill="#E4E7EC", width=1)
        d.text((x - 18, bottom + 14), str(xv), font=small_font, fill="#667085")
    points = []
    for th in np.linspace(2, 178, 700):
        c = abs(math.cos(math.radians(float(th))))
        val = math.sqrt((1 + c) / max(1 - c, 1e-12))
        val = min(val, 20)
        points.append((xmap(float(th)), ymap(val)))
    d.line(points, fill="#2E74B5", width=6)
    for angle in (15, 20, 30, 90):
        val = math.sqrt((1 + abs(math.cos(math.radians(angle)))) /
                        (1 - abs(math.cos(math.radians(angle)))))
        x, y = xmap(angle), ymap(val)
        d.ellipse((x - 8, y - 8, x + 8, y + 8), fill="#B38322")
        d.text((x + 12, y - 28), f"{angle}°: κ≈{val:.1f}", font=small_font, fill="#7A5A00")
    centered(d, (left, 830, right, 900), "两部雷达水平交叉波束角 θ（度）", body_font, "#20374C")
    d.text((15, 420), "条件数\nκ₂(A)", font=body_font, fill="#20374C", spacing=4, align="center")
    img.save(FIG_DIR / "beam_conditioning.png", dpi=(180, 180))

    # Figure 3: tensor structure and constraints.
    img = Image.new("RGB", (1890, 972), "white")
    d = ImageDraw.Draw(img)
    centered(d, (0, 15, 1890, 90), "时空风场张量与物理—几何约束", title_font, "#20374C")
    colors = ["#DCEAF7", "#C9DFEF", "#B5D3E8", "#9FC5E0"]
    for k in range(4):
        box = (120 + 55 * k, 270 - 45 * k, 690 + 55 * k, 720 - 45 * k)
        d.rounded_rectangle(box, radius=16, fill=colors[k], outline="#2E74B5", width=3)
    centered(d, (120, 150, 855, 240), "时空风场张量 𝒰", h_font, "#20374C")
    centered(d, (80, 790, 890, 860), "x × y × z × t × component(u,v,w)", body_font, "#667085")
    arrow(d, (870, 475), (1000, 475), width=6)
    items = [
        (1040, 180, 1810, 335, "观测一致性", "Σr ‖Wr⊙(Hr(𝒰)−yr)‖²"),
        (1040, 365, 1810, 520, "质量连续", "‖∇·(ρ𝒰)‖²"),
        (1040, 550, 1810, 705, "低秩结构", "Tucker / TT / tubal rank"),
        (1040, 735, 1810, 890, "时间与鲁棒项", "平流 / Huber / 稀疏异常"),
    ]
    for b in items:
        d.rounded_rectangle(b[:4], radius=16, fill="#F4F6F9", outline="#B38322", width=4)
        d.text((b[0] + 25, b[1] + 28), b[4], font=body_font, fill="#20374C")
        d.text((b[0] + 250, b[1] + 31), b[5], font=small_font, fill="#667085")
    img.save(FIG_DIR / "tensor_model.png", dpi=(180, 180))


def build_document():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    make_figures()

    doc = Document()
    setup_styles(doc)
    setup_header_footer(doc)
    bullet_id, number_id = add_numbering_definitions(doc)

    # Cover: narrative_proposal + editorial_cover header pattern.
    for _ in range(3):
        add_para(doc, "")
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(14)
    r = p.add_run("硕士论文研究与项目改造方案")
    set_run_font(r, size=11, bold=True, color=GOLD)

    p = doc.add_paragraph(style="Title")
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(12)
    r = p.add_run("交叉波束几何感知的物理约束\n低秩张量多普勒风场反演")
    set_run_font(r, size=28, bold=True, color=NAVY)

    p = doc.add_paragraph(style="Subtitle")
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(34)
    r = p.add_run("算法、模型、数学基础、数据需求、实验设计与现有项目改造路线")
    set_run_font(r, size=14, color=DARK_BLUE)

    add_callout(doc, "核心定位", "将现有双多普勒三维风反演系统提升为可投稿高水平期刊的广义低秩张量感知与PDE约束优化研究；PyDDA作为基线，不作为论文创新本身。", color=BLUE)

    for _ in range(2):
        add_para(doc, "")
    meta = doc.add_table(rows=4, cols=2)
    rows = [("作者", "________"), ("导师", "________"), ("专业方向", "数学 / 优化算法 / 张量计算"), ("版本日期", "2026年9月3日")]
    for i, (k, v) in enumerate(rows):
        meta.cell(i, 0).text = ""
        meta.cell(i, 1).text = ""
        p1 = meta.cell(i, 0).paragraphs[0]
        p1.alignment = WD_ALIGN_PARAGRAPH.RIGHT
        r = p1.add_run(k)
        set_run_font(r, size=10, bold=True, color=MUTED)
        p2 = meta.cell(i, 1).paragraphs[0]
        r = p2.add_run(v)
        set_run_font(r, size=10, color=BLACK)
    set_table_geometry(meta, [2700, 6660])
    # Remove visible metadata grid.
    for cell in [c for row in meta.rows for c in row.cells]:
        tc_pr = cell._tc.get_or_add_tcPr()
        borders = OxmlElement("w:tcBorders")
        for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
            node = OxmlElement(f"w:{edge}")
            node.set(qn("w:val"), "nil")
            borders.append(node)
        tc_pr.append(borders)

    doc.add_page_break()

    doc.add_heading("摘要", level=1)
    add_para(doc, "本方案面向数学专业优化算法方向研究生，围绕现有CINRAD双多普勒三维风场反演项目，提出一条能够支撑高水平期刊投稿的研究路线。核心问题不是把三维风场简单存成张量，而是把雷达径向速度观测、波束几何病态性、质量连续方程、时空相关性和局地中气旋结构统一建模为广义低秩张量反问题。建议主模型采用固定多线性秩Tucker表示，主算法采用带几何预条件的黎曼Gauss–Newton或黎曼共轭梯度法；以PyDDA 3DVAR为气象基线，以解析涡旋和数值模式风场构造OSSE真值，以真实连续体扫进行外部验证。论文应同时提供可辨识性或稳定性分析、算法收敛结果、复杂度分析、强基线对比、消融实验和可复现实验代码。")
    add_para(doc, "建议论文主标题为“Geometry-Aware Riemannian Optimization on Low-Rank Tensor Manifolds for Physics-Constrained Multi-Doppler Wind Retrieval”。偏数学投稿可面向Inverse Problems或SIAM Journal on Scientific Computing；偏计算物理可面向Journal of Computational Physics；若理论深度不足但雷达检验充分，则优先考虑Journal of Atmospheric and Oceanic Technology、Atmospheric Measurement Techniques或IEEE Transactions on Geoscience and Remote Sensing。")

    doc.add_heading("关键词", level=2)
    add_para(doc, "双多普勒雷达；三维风场反演；低秩张量；Tucker分解；张量列车；黎曼优化；PDE约束优化；交叉波束角；病态反问题；中气旋")

    doc.add_heading("目录", level=1)
    toc_p = doc.add_paragraph()
    add_field(toc_p, ' TOC \\o "1-3" \\h \\z \\u ', "请在Word中按 Ctrl+A、F9 更新目录")
    doc.add_page_break()

    doc.add_heading("1 研究定位与拟解决问题", level=1)
    doc.add_heading("1.1 现有项目能够完成什么", level=2)
    add_para(doc, "当前项目已经形成从CINRAD WRS RSTM产品读取、PPI体扫合并、速度质量控制、统一笛卡尔网格化、双多普勒直接合成初猜、PyDDA三维变分反演，到NetCDF输出和PyVista三维流线显示的完整链条。核心状态量为三维网格上的u、v、w，并输出水平风速、散度、垂直涡度、交叉波束角、双雷达有效掩膜、径向速度回代残差和质量连续残差。")
    add_para(doc, "项目近期已增加时间配对限制、交叉波束角对称传递、反射率门槛、垂直速度与连续方程质量门控、PROJ路径修复以及合成涡旋闭环测试。这些工作使项目成为可信的科研基线，但还不是新算法论文：目前主求解器仍是PyDDA的L-BFGS-B，权重主要人工给定，中气旋检测仍主要依赖反演风场涡度柱，且真实案例数量不足。")

    doc.add_heading("1.2 当前案例暴露出的数学问题", level=2)
    add_bullet(doc, "几何病态：M产品中心处两部雷达交叉波束角约14°–15°，局地投影矩阵接近秩亏。", bullet_id)
    add_bullet(doc, "结构缺测：雷达盲区和波束覆盖造成的是成片、成层和扇区式缺测，不满足经典随机张量补全假设。", bullet_id)
    add_bullet(doc, "垂直速度弱可观：低仰角径向速度对w的信息很少，w主要由质量连续、边界和背景场间接决定。", bullet_id)
    add_bullet(doc, "正则化偏差：强平滑会抹去局地涡旋，弱平滑又会放大噪声和几何误差。", bullet_id)
    add_bullet(doc, "时间非同步：不同雷达体扫和各径向采样时间不同，快速演变风暴不满足严格稳态。", bullet_id)
    add_bullet(doc, "验证困难：业务M产品是比较基线而非真实三维风真值；必须引入OSSE、留出雷达或独立观测。", bullet_id)

    doc.add_heading("1.3 高水平期刊所要求的研究增量", level=2)
    add_callout(doc, "判断标准", "仅把Tucker或TT调用到现有数据上通常不够。好期刊需要一个可推广到其他矢量层析、PDE反演或多传感器融合问题的新模型/新算法，并用雷达问题证明其价值。", color=RED)
    number_id = restart_numbering(doc, number_id)
    add_number(doc, "模型创新：建立“投影观测 + 结构缺测 + 低秩流形 + PDE弱约束”的统一张量感知模型。", number_id)
    add_number(doc, "算法创新：构造几何感知预条件的黎曼算法、秩自适应机制或可证明的分裂算法。", number_id)
    add_number(doc, "理论创新：给出可辨识性条件、稳定性误差界、收敛性或局部收敛速度。", number_id)
    add_number(doc, "实验完整性：解析OSSE、数值模式OSSE、真实雷达、多噪声多缺测、强基线、消融和统计置信区间。", number_id)
    add_number(doc, "可复现性：固定数据划分、公开配置、随机种子、环境锁定、运行日志和生成论文图表的脚本。", number_id)

    add_figure(doc, FIG_DIR / "workflow.png", "图1  建议采用的观测—反演—验证双轨研究架构", alt="雷达观测经质量控制和观测算子进入张量反演，PyDDA、OSSE和真实过程构成三条验证支路。")

    doc.add_heading("2 现有算法及其数学基础", level=1)
    doc.add_heading("2.1 雷达径向速度观测方程", level=2)
    add_para(doc, "设笛卡尔坐标中风矢量为V(x,y,z,t)=(u,v,w)ᵀ，第r部雷达指向格点的单位波束向量为eᵣ=(cosφ sinα, cosφ cosα, sinφ)ᵀ，其中α为方位角、φ为仰角。考虑水凝物终端下落速度wₜ后，径向速度近似满足：")
    add_equation(doc, "Vᵣ = cosφ·sinα·u + cosφ·cosα·v + sinφ·(w − |wₜ|) + εᵣ", "(1)")
    add_para(doc, "给定雷达位置和网格坐标后，式(1)对风场是线性的，因此可构造矩阵无关的正向算子Hᵣ及其伴随Hᵣ*。真实算法中不要显式形成百万维稀疏矩阵，而应实现forward(U)和adjoint(Y)，并通过内积一致性测试验证伴随正确性。")

    doc.add_heading("2.2 双雷达直接水平风合成", level=2)
    add_para(doc, "忽略低仰角下w项时，两部雷达在一个格点给出二维线性系统A[u,v]ᵀ=b。现有src/dual_radar_retrieval.py逐格求解该系统，并由连续方程积分得到w，作为PyDDA初猜。直接方法速度快、可解释，但误差随A的条件数放大，且垂直积分会累积水平散度误差。")
    add_equation(doc, "A = [[cosφ₁sinα₁, cosφ₁cosα₁], [cosφ₂sinα₂, cosφ₂cosα₂]],   b = [Vᵣ₁, Vᵣ₂]ᵀ", "(2)")

    doc.add_heading("2.3 PyDDA 3DVAR基线", level=2)
    add_para(doc, "现有src/retrieval.py调用PyDDA，将观测、背景、非弹性质量连续和平滑项组合为代价函数，并用L-BFGS-B优化。其简化形式为：")
    add_equation(doc, "J(U)=CₒJₒ(U)+CᵦJᵦ(U)+CₘJₘ(U)+CₓJₓ(U)+CᵧJᵧ(U)+C_zJ_z(U)", "(3)")
    add_bullet(doc, "观测项Jₒ：反演风投影回雷达方向后与实测径向速度的加权平方误差。", bullet_id)
    add_bullet(doc, "背景项Jᵦ：分析风与VWP、探空或数值模式背景风的偏差。", bullet_id)
    add_bullet(doc, "质量连续项Jₘ：约束∇·(ρV)接近零，以间接确定w。", bullet_id)
    add_bullet(doc, "平滑项：约束各方向二阶导数，抑制格点尺度噪声。", bullet_id)
    add_para(doc, "PyDDA是可靠基线而非论文创新。论文需要完整记录其版本、权重、初猜、边界、过滤窗口和交叉角阈值，并将最终结果按同一有效掩膜与新方法公平比较。")

    doc.add_heading("2.4 中气旋检测与三维显示", level=2)
    add_para(doc, "当前visualize_3d.py在双雷达有效区寻找气旋式垂直涡度柱，叠加反射率、垂直连续性和环流/流线一致性筛选。该模块应继续作为结果诊断和可视化，不应参与风场求解器的调参目标。业务M产品来自单雷达径向速度切变对象，与三维风场涡度是不同证据链；论文需要分别报告两者。")

    doc.add_heading("3 几何病态性与反问题分析", level=1)
    doc.add_heading("3.1 交叉波束角与条件数", level=2)
    add_para(doc, "在忽略仰角差且两个波束向量归一化的理想二维情形，两个观测方向夹角为θ时，AᵀA的特征值为1±|cosθ|，因此二范数条件数为：")
    add_equation(doc, "κ₂(A)=sqrt((1+|cosθ|)/(1−|cosθ|))", "(4)")
    add_para(doc, "θ接近0°或180°时条件数发散；θ=90°时最优。按式(4)，15°时κ约7.6、20°时约5.7、30°时约3.7。实际三维问题还叠加低仰角对w不敏感、插值和平滑误差，因此病态程度通常更严重。")
    add_figure(doc, FIG_DIR / "beam_conditioning.png", "图2  双雷达交叉波束角与局地反演条件数的关系", width=4.3, alt="条件数在交叉角接近0度或180度时快速增大，在90度达到最小值1。")

    doc.add_heading("3.2 可观测子空间和不可观测子空间", level=2)
    add_para(doc, "观测算子H只约束风场在雷达波束方向上的投影。即使每个格点都有两部雷达，三维向量仍少一个直接观测分量；缺测区则秩进一步下降。背景、连续方程、时间演化和低秩先验的作用，是在不可观测子空间中选择一个物理上合理的解，而不是创造新的独立观测。因此论文必须同步输出不确定度或可辨识性指标，不能把正则化结果等同于真值。")

    doc.add_heading("3.3 PDE约束", level=2)
    add_para(doc, "对流尺度常采用非弹性近似：")
    add_equation(doc, "∇·(ρ₀V)=0   ⇔   ∂u/∂x+∂v/∂y+∂w/∂z+w·∂lnρ₀/∂z=0", "(5)")
    add_para(doc, "在离散网格上定义DρU表示该残差。可将其作为弱约束μ‖DρU‖²/2，也可在切空间内进行等式约束优化。弱约束更能容纳离散误差和模型误差，适合作为第一篇论文的主方案。后续可加入垂直涡度方程，但它依赖时间导数和平流速度，体扫时间较长时模型误差可能增加。")

    doc.add_heading("4 张量表示与低秩先验", level=1)
    doc.add_heading("4.1 状态张量", level=2)
    add_para(doc, "将连续K个体扫组成五阶张量U∈R^(nₓ×nᵧ×n_z×K×3)。空间、时间和分量模式具有不同物理意义，不能随意把所有维度摊平成矩阵。时间维能够利用风暴结构短时连续性；高度维描述垂直相关；分量维耦合u、v、w。若只有一个时次，低秩假设证据明显变弱，因此高水平论文应优先研究时空反演。")
    add_figure(doc, FIG_DIR / "tensor_model.png", "图3  时空风场张量及四类约束", alt="五阶时空风场张量通过观测一致性、质量连续、低秩结构和时间鲁棒约束联合反演。")

    doc.add_heading("4.2 分解形式选择", level=2)
    add_table(doc,
              ["模型", "表示", "优点", "主要风险", "本研究建议"],
              [
                  ("CP", "秩一项之和", "参数少、解释直接", "最佳低秩逼近可能不稳定；秩选择困难", "仅作对比"),
                  ("Tucker", "核心张量×各模因子矩阵", "适合5阶数据；HOSVD易初始化；多线性秩清晰", "核心规模随阶数增长", "主模型"),
                  ("TT", "张量列车核心链", "高阶存储近线性；适合长时间窗", "模式顺序影响大；实现复杂", "扩展模型"),
                  ("t-SVD", "沿指定模的卷积/傅里叶分解", "有张量核范数和近端算子", "天然偏三阶；重排可能破坏物理模式", "时间模对比"),
              ], [1050, 2100, 2100, 2500, 1610], font_size=8.4)
    add_para(doc, "推荐先采用Tucker多线性秩(rₓ,rᵧ,r_z,r_t,r_c)，因为它与空间—高度—时间—分量模式自然对应，能够采用HOSVD初始化，并可在固定秩流形上进行黎曼优化。TT适合后续扩大时间窗或引入多参数维度。")

    doc.add_heading("4.3 低秩假设必须先检验", level=2)
    number_id = restart_numbering(doc, number_id)
    add_number(doc, "将连续体扫的PyDDA风场、径向速度和反射率分别组成张量。", number_id)
    add_number(doc, "计算各模展开矩阵的奇异值衰减和累计能量，不只报告压缩率。", number_id)
    add_number(doc, "在不同天气阶段、不同空间子域和不同变量上比较有效秩。", number_id)
    add_number(doc, "检查低秩截断后最大涡度、环流和极值风速的保持率，防止总体RMSE较小但中气旋被抹平。", number_id)
    add_number(doc, "若全域低秩不成立，改用风暴对象中心化、滑动块Tucker或低秩背景加局地涡旋分解。", number_id)

    doc.add_heading("4.4 为什么不能直接套用经典张量补全", level=2)
    add_para(doc, "经典张量补全通常研究从随机已知元素恢复低秩张量，并依赖不相干性等条件。本项目的未知量U没有被直接采样，观测是H(U)；而且缺测由雷达波束和地形决定，具有强结构性。因此应将问题表述为generalized tensor sensing / low-rank tensor inverse problem，并分析复合算子PΩH在低秩流形切空间上的限制可逆性。")

    doc.add_heading("5 建议的主模型", level=1)
    doc.add_heading("5.1 几何感知的物理约束Tucker模型", level=2)
    add_equation(doc, "min_{U∈M_r}  ½Σᵣ‖Wᵣ⊙PΩᵣ(Hᵣ(U)−yᵣ)‖²_F + μ/2‖DρU‖²_F + γ/2‖T(U)‖²_F + ηR_rob(U)", "(6)")
    add_para(doc, "M_r表示固定Tucker多线性秩流形；Dρ为质量连续算子；T为时间差分或半拉格朗日平流残差；R_rob可取Huber损失、总变分或稀疏异常项。第一篇论文建议控制复杂度：主模型使用Tucker秩、加权Huber观测项、质量连续和一阶时间差分；其他项放在扩展实验。")

    doc.add_heading("5.2 几何权重", level=2)
    add_para(doc, "将二值交叉角阈值改为连续权重。令s(x)=σmin(A(x))/σmax(A(x))，并结合信噪比、距离、插值覆盖和速度退模糊质量构造：")
    add_equation(doc, "Wᵣ(x)=mᵣ(x)·q_BCA(x)·q_SNR(x)·q_range(x)·q_QC(x)", "(7)")
    add_para(doc, "其中mᵣ为有效掩膜。q_BCA可以采用截断sinθ、条件数倒数或从观测误差协方差推导。论文需要比较硬阈值、sinθ权重、条件数权重和学习型权重，并检查校准后不确定度是否随交叉角合理增大。")

    doc.add_heading("5.3 鲁棒观测模型", level=2)
    add_para(doc, "速度退模糊失败和孤立异常值会使平方损失产生过大影响。可采用Huber损失：小残差保持二次，大残差转为一次；或显式引入稀疏误差S：")
    add_equation(doc, "min_{U∈M_r,S}  ½‖W⊙(H(U)+S−Y)‖²_F + λ_S‖S‖₁ + μ/2‖DρU‖²_F + γR_t(U)", "(8)")
    add_para(doc, "如采用S变量，ADMM或交替最小化更自然；如论文主攻黎曼算法，则优先采用光滑Huber近似，避免同时引入过多算法模块。")

    doc.add_heading("5.4 局地涡旋保持", level=2)
    add_para(doc, "低秩正则容易压低尖锐旋转。建议至少采用一种保护机制：对风暴对象做局地块张量；采用多尺度Tucker；将环境风L与局地扰动V分解；或在目标函数中加入涡度保持/垂直涡度方程弱约束。任何机制都应通过涡度峰值保持率、环流误差和中心位置误差验证，而不能只看全域RMSE。")

    doc.add_heading("6 优化算法设计", level=1)
    doc.add_heading("6.1 主算法：几何预条件黎曼Gauss–Newton", level=2)
    add_para(doc, "固定Tucker秩集合在满秩条件下可视为光滑流形。将目标函数限制在M_r上，每一步先计算欧氏梯度，再投影到当前点切空间，使用近似Gauss–Newton Hessian求搜索方向，最后通过retraction返回流形。")
    add_equation(doc, "grad f(U)=P_{T_U M_r}(∇f(U)),   U_{k+1}=R_{U_k}(α_k ξ_k)", "(9)")
    add_para(doc, "矩阵无关H/H*、Dρ/Dρ*和时间算子使梯度计算无需形成大矩阵。预条件器可近似H*W²H+μDρ*Dρ+γT*T，并用局地交叉角条件数调整尺度，从而降低好几何区和差几何区之间的曲率差异。")

    doc.add_heading("6.2 推荐迭代流程", level=2)
    steps = [
        "用背景场或PyDDA结果做HOSVD，得到初始Tucker因子和核心张量。",
        "计算各雷达径向速度预测、鲁棒残差和几何权重。",
        "调用H*累积观测梯度，加入质量连续和时间约束梯度。",
        "将欧氏梯度投影到Tucker切空间。",
        "用预条件共轭梯度近似求解切空间Gauss–Newton方程。",
        "采用Armijo线搜索或信赖域接受步长，通过截断HOSVD完成retraction。",
        "记录目标函数各分项、黎曼梯度范数、秩、时间、内存和留出误差。",
        "满足梯度范数、相对目标下降和最大迭代数的联合条件后停止。",
    ]
    number_id = restart_numbering(doc, number_id)
    for s in steps:
        add_number(doc, s, number_id)

    doc.add_heading("6.3 备选算法：ADMM/近端交替最小化", level=2)
    add_para(doc, "若采用张量核范数与稀疏异常的凸松弛，可以引入辅助变量，把观测一致性、低秩近端、连续方程和稀疏项拆分，用ADMM求解。优点是各子问题清晰、便于并行；缺点是高阶张量核范数定义不唯一、每轮多次SVD开销大，且雷达复合算子子问题仍需迭代线性求解。建议将ADMM作为对比或第二篇工作。")

    doc.add_heading("6.4 秩选择与秩自适应", level=2)
    add_bullet(doc, "初始秩：根据各模奇异值累计能量和噪声水平设定，不使用测试集。", bullet_id)
    add_bullet(doc, "验证秩：使用成块留出径向速度，而不是随机留出相邻格点，避免空间泄漏。", bullet_id)
    add_bullet(doc, "秩增长：当切空间梯度在法空间仍有显著分量时增加相应模式秩。", bullet_id)
    add_bullet(doc, "秩压缩：定期截断小奇异值，并验证涡度和环流没有被显著削弱。", bullet_id)

    doc.add_heading("6.5 可证明的理论结果", level=2)
    add_callout(doc, "高水平期刊关键", "论文不要预先承诺全局最优。固定秩问题是非凸的，更现实的目标是建立可辨识性、噪声稳定性、收敛到一阶稳定点，以及在良好初始化和局部限制强凸条件下的局部线性/超线性收敛。", color=GOLD)
    add_table(doc,
              ["理论问题", "建议结论", "主要工具", "验证方式"],
              [
                  ("存在性", "有界可行集或正则化下极小点存在", "紧性、下半连续性", "给出假设与证明"),
                  ("局部可辨识", "PΩH在T_U M_r上单射或满足限制稳定性", "切空间、最小奇异值", "数值谱估计"),
                  ("误差界", "误差受噪声、低秩逼近误差和几何常数控制", "扰动分析、RSC/RIP类条件", "OSSE斜率验证"),
                  ("算法收敛", "黎曼梯度范数趋零；信赖域全局收敛到稳定点", "黎曼优化标准理论", "收敛曲线"),
                  ("局部速度", "良好初始化下线性或超线性收敛", "Gauss–Newton局部分析", "不同初值实验"),
                  ("复杂度", "存储和每轮计算随秩而非全张量规模增长", "张量收缩计数", "时间/内存扩展实验"),
              ], [1400, 3200, 2400, 2360], font_size=8.4)

    doc.add_heading("7 所需数学知识体系", level=1)
    add_table(doc,
              ["知识模块", "必须掌握内容", "在论文中的用途"],
              [
                  ("多线性代数", "模乘、展开、Kronecker/Khatri–Rao积、HOSVD、Tucker/TT秩", "状态压缩、梯度和复杂度"),
                  ("数值线性代数", "SVD、最小二乘、条件数、Krylov方法、预条件", "几何病态与内层线性求解"),
                  ("非凸优化", "一阶条件、线搜索、信赖域、KL性质、块坐标法", "收敛分析"),
                  ("黎曼优化", "切空间、投影、retraction、向量传输、黎曼Hessian", "固定秩张量主算法"),
                  ("反问题", "可辨识性、Tikhonov正则、稳定性、偏差—方差", "误差界和不确定度"),
                  ("PDE约束优化", "弱约束、伴随算子、离散化一致性、边界条件", "质量连续与涡度方程"),
                  ("统计学习", "交叉验证、Bootstrap、校准、事件级数据划分", "模型选择和显著性"),
                  ("雷达气象", "径向速度、退模糊、波束传播、终端速度、体扫", "观测模型不犯物理错误"),
              ], [1700, 4300, 3360], font_size=8.5)

    doc.add_heading("8 数据需求与数据组织", level=1)
    doc.add_heading("8.1 当前已有数据", level=2)
    add_para(doc, "现有项目包含Z9317–Z9543的2024年过程和Z9539–Z9532的2026年过程，具有速度PPI、反射率PPI、部分VWP、M/TVS及其他二级产品，并已产生若干三维风场NetCDF。这些数据足够完成代码原型、低秩可行性分析和真实案例演示，但不足以支撑高水平期刊关于泛化能力的结论。")

    doc.add_heading("8.2 建议新增数据", level=2)
    add_table(doc,
              ["数据", "最低用途", "推荐要求", "注意事项"],
              [
                  ("双/多雷达基数据", "主要观测", "连续体扫、统一标定、包含逐径向时间", "优先Level-II/基数据而非仅产品标称时刻"),
                  ("反射率与双偏振量", "下落速度、QC、风暴对象", "与速度同体扫或精确配对", "不能用任意时次反射率替代"),
                  ("VWP/探空/模式", "背景场", "记录来源、空间时间插值", "背景误差不能视为零"),
                  ("第三部雷达", "独立验证", "尽量不参与反演，用于留出投影检验", "最有价值的真实三维约束"),
                  ("风廓线/测风激光雷达", "局地风验证", "时间高度匹配", "代表性尺度不同"),
                  ("地面站", "近地层验证", "高频风速风向", "只能验证最低层"),
                  ("M/TVS/人工判识", "中气旋对象验证", "两名以上人员独立复核", "M产品不是绝对真值"),
                  ("数值模式真值", "模式OSSE", "高分辨率三维风和微物理场", "需模拟雷达观测误差"),
              ], [1500, 1800, 3200, 2860], font_size=8.2)

    doc.add_heading("8.3 样本规模与划分", level=2)
    add_para(doc, "建议覆盖至少一个完整强对流季，包含几十个中气旋过程及足量非中气旋对照，且每个过程包含连续多个体扫。这里的数量是研究设计建议，不是期刊硬性规定；最终应根据事件独立性和置信区间宽度进行功效评估。训练/调参/测试必须按天气过程或日期划分，不能把同一风暴相邻体扫分到不同集合。")
    add_bullet(doc, "开发集：选择少量案例调试物理算子和优化器。", bullet_id)
    add_bullet(doc, "验证集：选择秩、正则参数、停止条件和质量门槛。", bullet_id)
    add_bullet(doc, "封闭测试集：只在最终模型确定后运行一次。", bullet_id)
    add_bullet(doc, "跨站点测试：使用未参与开发的雷达组合检验几何泛化。", bullet_id)

    doc.add_heading("8.4 建议数据格式", level=2)
    add_para(doc, "内部研究数据建议使用xarray Dataset，并按case_id组织Zarr或NetCDF。坐标至少包含time、z、y、x、component和radar；变量包括radial_velocity、azimuth、elevation、fall_speed、reflectivity、valid_mask、bca、quality_weight、u/v/w背景场及人工标签。所有变量必须记录单位、缺测值、站点坐标、扫描时刻、VCP和处理版本。")

    doc.add_heading("9 实验过程", level=1)
    doc.add_heading("9.1 阶段0：冻结和复现现有基线", level=2)
    number_id = restart_numbering(doc, number_id)
    add_number(doc, "将当前目录纳入Git版本控制，锁定Python和依赖版本。", number_id)
    add_number(doc, "保存两套代表性输出、命令行参数、日志和NetCDF元数据。", number_id)
    add_number(doc, "统一PyDDA、直接双雷达和张量方法使用的网格、掩膜、反射率订正和评价区域。", number_id)
    add_number(doc, "把已有10项测试作为回归门槛；任何新模块不得改变基线结果。", number_id)

    doc.add_heading("9.2 阶段1：低秩经验验证", level=2)
    number_id = restart_numbering(doc, number_id)
    add_number(doc, "形成连续体扫时空张量，分别对全域、双雷达有效域和风暴对象子域做HOSVD。", number_id)
    add_number(doc, "绘制每一模式奇异值谱、累计能量、压缩率和随风暴阶段变化的有效秩。", number_id)
    add_number(doc, "评估不同秩截断对风速RMSE、最大涡度、环流和质量连续残差的影响。", number_id)
    add_number(doc, "若时间模秩不稳定，采用风暴平移订正或随风暴移动的坐标系后重新检验。", number_id)

    doc.add_heading("9.3 阶段2：解析OSSE", level=2)
    add_para(doc, "构造具有已知u、v、w的三维解析风场，例如Rankine涡旋、Burgers–Rott涡旋、倾斜涡旋柱、旋转上升流、辐合—辐散偶极以及叠加环境切变。将真值投影到实际雷达位置，加入终端下落速度、噪声、缺测和时间平移，再用各方法恢复。")
    add_table(doc,
              ["因素", "推荐水平"],
              [
                  ("交叉角", "10°, 15°, 20°, 30°, 45°, 60°, 90°"),
                  ("速度噪声标准差", "0, 1, 2, 4 m/s"),
                  ("结构缺测比例", "0%, 10%, 30%, 50%（扇区/整层/低层缺测）"),
                  ("双站时间差", "0, 30, 60, 120 s"),
                  ("涡旋结构", "轴对称、倾斜、非轴对称、多涡旋、强环境切变"),
                  ("随机重复", "每个代表组合至少5–10个噪声种子"),
              ], [2700, 6660], font_size=9)
    add_para(doc, "完整笛卡尔积可能计算量过大，可先做筛选实验，再对关键交叉角和噪声水平进行完整重复。所有方法使用同一模拟观测，报告均值、标准差和事件级Bootstrap置信区间。")

    doc.add_heading("9.4 阶段3：数值模式OSSE", level=2)
    add_para(doc, "从高分辨率对流模式或公开超单体模拟中提取四维风、反射率和水凝物场，使用雷达模拟器生成非规则观测。这一步比解析涡旋更接近真实复杂流动，可检验低秩模型是否在多尺度湍流、冷池和非轴对称中气旋下仍有效。应把用于生成伪观测的完整模式风场保留为真值。")

    doc.add_heading("9.5 阶段4：真实雷达检验", level=2)
    number_id = restart_numbering(doc, number_id)
    add_number(doc, "先在开发案例检查数据读取、QC、观测回代和质量门控。", number_id)
    add_number(doc, "若有三部雷达，用两部反演、第三部完全留出，比较预测径向速度。", number_id)
    add_number(doc, "若只有两部雷达，采用成块留出：留出完整径向、扇区、仰角或时间段，禁止随机格点留出。", number_id)
    add_number(doc, "与PyDDA、直接双雷达、无低秩3DVAR、矩阵低秩和不同张量模型比较。", number_id)
    add_number(doc, "对中气旋对象报告中心误差、环流、垂直连续性、首次识别时间和跟踪连续率。", number_id)
    add_number(doc, "由人工判识人员盲评风场和中气旋对象，避免只依赖业务M产品。", number_id)

    doc.add_heading("9.6 阶段5：消融与敏感性实验", level=2)
    add_table(doc,
              ["实验", "去除/替换项", "回答的问题"],
              [
                  ("A1", "去掉低秩约束", "增益来自张量结构还是普通3DVAR？"),
                  ("A2", "去掉质量连续", "物理约束对w和泛化的贡献？"),
                  ("A3", "硬BCA阈值替换连续权重", "几何权重是否真正改善病态区？"),
                  ("A4", "平方损失替换Huber", "鲁棒项是否抵抗退模糊异常？"),
                  ("A5", "单时次替换时空窗", "时间张量结构提供多少信息？"),
                  ("A6", "Tucker替换矩阵/CP/TT/t-SVD", "收益是否来自特定张量模型？"),
                  ("A7", "普通梯度替换几何预条件", "算法加速是否来自预条件？"),
                  ("A8", "不同初始化", "非凸优化对初值是否敏感？"),
              ], [950, 3600, 4810], font_size=8.6)

    doc.add_heading("10 评价指标与统计设计", level=1)
    doc.add_heading("10.1 风场恢复指标", level=2)
    add_bullet(doc, "真值RMSE/MAE：分别对u、v、w计算，并按高度、BCA、距离和回波强度分层。", bullet_id)
    add_bullet(doc, "向量误差：风速偏差、风向误差、三维向量夹角。", bullet_id)
    add_bullet(doc, "留出观测误差：第三雷达或成块留出径向速度RMSE。", bullet_id)
    add_bullet(doc, "物理残差：质量连续残差P50/P95/P99、边界通量和能量谱。", bullet_id)
    add_bullet(doc, "结构指标：涡度峰值保持率、环流误差、上升气流体积和中心位置误差。", bullet_id)
    add_bullet(doc, "计算指标：总耗时、单次迭代耗时、峰值内存、迭代次数和扩展效率。", bullet_id)

    doc.add_heading("10.2 中气旋对象指标", level=2)
    add_equation(doc, "POD=H/(H+M),   FAR=F/(H+F),   CSI=H/(H+M+F)", "(10)")
    add_para(doc, "H、M、F分别为命中、漏报和空报。还应报告对象中心距离、垂直重叠率、生命周期重叠率、首次识别提前量和ID切换次数。阈值必须在验证集确定；测试集只做最终报告。")

    doc.add_heading("10.3 不确定度与校准", level=2)
    add_para(doc, "高水平论文最好不仅给点估计，还给可靠度。可用Gauss–Newton Hessian的低秩近似、Laplace近似、Bootstrap或多初值集合估计方差。检查预测区间覆盖率、区间宽度与BCA的关系，以及质量等级与实际误差是否单调。若暂时不能完成完整概率反演，至少输出局地可辨识性指标σmin、条件数和留出残差。")

    doc.add_heading("11 怎样修改当前项目", level=1)
    doc.add_heading("11.1 改造原则", level=2)
    add_bullet(doc, "保留现有run_retrieval.py与WindRetrieval，作为可重复PyDDA基线。", bullet_id)
    add_bullet(doc, "新增张量求解器，不在PyDDA函数内部打补丁，保证公平对比。", bullet_id)
    add_bullet(doc, "把数据层、物理算子、张量模型、优化器、实验和可视化解耦。", bullet_id)
    add_bullet(doc, "所有算子必须同时实现forward和adjoint，并有独立数学测试。", bullet_id)
    add_bullet(doc, "论文图表必须由实验记录自动生成，禁止手工抄数。", bullet_id)

    doc.add_heading("11.2 建议新增目录和模块", level=2)
    add_table(doc,
              ["新增文件/目录", "职责", "关键接口或输出"],
              [
                  ("configs/", "实验配置和数据划分", "YAML：模型、秩、权重、随机种子"),
                  ("src/tensor_data.py", "连续体扫组装、标准化、掩膜", "WindTensorCase / xarray Dataset"),
                  ("src/observation_operator.py", "雷达投影算子及伴随", "forward(U), adjoint(R), dot_test()"),
                  ("src/physics_operators.py", "连续、梯度、旋度、时间算子", "Dρ, Dρ*, T, T*"),
                  ("src/tensor_models.py", "Tucker/TT表示与HOSVD初始化", "reconstruct(), compress(), rank"),
                  ("src/tensor_objective.py", "统一目标函数与分项梯度", "value_and_grad(), diagnostics"),
                  ("src/riemannian_solver.py", "主优化器", "solve(), history, convergence status"),
                  ("src/preconditioners.py", "BCA/物理感知预条件", "apply_preconditioner()"),
                  ("src/rank_selection.py", "秩选择与增长/截断", "select_rank(), adapt_rank()"),
                  ("src/uncertainty.py", "条件数、Hessian近似、不确定度", "quality tensor"),
                  ("src/osse.py", "解析和模式伪观测", "truth, simulated Vr, masks"),
                  ("experiments/", "批量实验与消融", "结构化CSV/Parquet + NetCDF"),
                  ("tests_tensor/", "算子、梯度、恢复和回归测试", "pytest测试集"),
              ], [2500, 2900, 3960], font_size=8.2)

    doc.add_heading("11.3 对现有文件的修改", level=2)
    add_table(doc,
              ["现有文件", "建议修改", "验收标准"],
              [
                  ("run_retrieval.py", "增加--method pydda|tucker-riemannian、--config、--time-window、--rank和正则参数；公共输入流程只执行一次", "同一案例可一键运行所有基线"),
                  ("src/retrieval.py", "抽取网格和输出公共函数；WindRetrieval仅负责PyDDA；禁止新算法复用其隐式状态", "PyDDA旧结果回归一致"),
                  ("src/dual_radar_retrieval.py", "暴露局地观测矩阵、σmin、σmax和条件数；保留直接解作为初猜", "理论图和算法权重来源一致"),
                  ("src/pipeline.py", "保留逐径向时间、QC标志和观测误差；避免过早填零和过度笛卡尔插值", "缺测与异常可追踪"),
                  ("visualize_3d.py", "读取方法名、置信度和不确定度；MDA检测与三维风确认分层显示", "差几何区不显示为可信三维风"),
                  ("requirements.txt", "增加研究环境文件；候选依赖tensorly、pymanopt/torch、dask、zarr、pytest", "全新环境可复现"),
                  ("README.md", "增加数学模型、配置、数据版本、复现实验和论文结果索引", "审稿人能按命令复现主表"),
              ], [1800, 5000, 2560], font_size=8.2)

    doc.add_heading("11.4 第一批必须增加的测试", level=2)
    number_id = restart_numbering(doc, number_id)
    add_number(doc, "伴随测试：|⟨HU,R⟩−⟨U,H*R⟩|/尺度 < 10⁻⁸。", number_id)
    add_number(doc, "梯度测试：解析/自动微分梯度与中心有限差分方向导数相对误差 < 10⁻⁶。", number_id)
    add_number(doc, "连续算子测试：构造解析无散风，检查DρU接近离散截断误差。", number_id)
    add_number(doc, "几何测试：交叉角趋近0°时σmin下降、条件数上升，90°附近最优。", number_id)
    add_number(doc, "低秩恢复测试：无噪声充分观测下恢复已知Tucker低秩张量。", number_id)
    add_number(doc, "鲁棒测试：加入稀疏异常后，Huber/稀疏模型优于平方损失。", number_id)
    add_number(doc, "回归测试：method=pydda时现有10项测试及代表NetCDF指标保持一致。", number_id)

    doc.add_heading("11.5 推荐命令行形态", level=2)
    add_callout(doc, "示例", "python run_retrieval.py --method tucker-riemannian --config configs/paper_main.yaml --station1 Z9539 --station2 Z9532 --time-window 6 --rank 30,30,8,3,3", color=GREEN)
    add_para(doc, "实际实现时应让配置文件成为唯一参数源，命令行只覆盖个别字段。每次运行生成run_id目录，保存resolved_config.yaml、环境信息、Git提交号、目标函数历史、质量指标和输出NetCDF。当前目录尚无Git元数据，正式研究前应先建立版本控制或迁移到受控仓库。")

    doc.add_heading("12 面向好期刊的投稿设计", level=1)
    doc.add_heading("12.1 期刊路线", level=2)
    add_table(doc,
              ["期刊路线", "论文必须突出", "不够的内容", "建议定位"],
              [
                  ("Inverse Problems", "可辨识性、稳定性、正则化理论及广义雷达反问题", "只有新损失函数和案例", "理论最强、风险最高"),
                  ("SIAM J. Scientific Computing", "可推广数值算法、理论/强启发、复杂度、可复现大规模实验", "只调用现成张量库", "主攻数学计算"),
                  ("Journal of Computational Physics", "物理建模+创新数值方法+鲁棒性和复杂度", "只有气象产品比较", "数学和物理均衡"),
                  ("IEEE TGRS", "遥感观测模型、完整真实数据、显著方法增量", "只有OSSE或单案例", "偏遥感算法"),
                  ("JTECH / AMT", "雷达反演方法、观测细节、充分验证和业务意义", "理论很强但气象验证弱", "较稳妥的优质领域期刊"),
                  ("SIAM J. Optimization", "对广泛优化问题有实质新理论，而非雷达专用技巧", "应用型黎曼求解器", "除非理论突破，否则不首投"),
              ], [2000, 3300, 2200, 1860], font_size=8.2)
    add_para(doc, "SISC官方强调新数值方法需要有理论或强启发支撑，并用有意义的计算结果证明有效性；Inverse Problems要求对反问题领域具有实质性推进；JCP强调方法的有效性、鲁棒性、复杂度和可复现性；JTECH与AMT更看重遥感反演、测量方法和验证。选刊时不要只看影响因子，应在模型完成后根据“理论贡献与气象验证的比例”决定。")

    doc.add_heading("12.2 推荐的论文核心贡献表述", level=2)
    number_id = restart_numbering(doc, number_id)
    add_number(doc, "提出一种面向非随机结构缺测和方向投影观测的物理约束低秩张量感知模型。", number_id)
    add_number(doc, "建立交叉波束角、复合观测算子切空间最小奇异值与反演稳定性的联系。", number_id)
    add_number(doc, "提出几何感知预条件的Tucker流形Gauss–Newton算法，并给出收敛与复杂度分析。", number_id)
    add_number(doc, "通过解析OSSE、模式OSSE和真实多雷达资料证明算法在差几何、缺测和异常值下优于3DVAR及低秩基线。", number_id)
    add_number(doc, "公开可复现实现，并为每个格点输出可辨识性/不确定度，而不是无条件填充风场。", number_id)

    doc.add_heading("12.3 论文结构建议", level=2)
    add_table(doc,
              ["章节", "主要内容", "页数建议"],
              [
                  ("1 Introduction", "问题、相关工作、差距、贡献", "2–3"),
                  ("2 Forward problem", "雷达算子、几何病态、PDE约束", "3–4"),
                  ("3 Tensor model", "Tucker流形、目标函数、权重和可辨识性", "4–5"),
                  ("4 Algorithm", "预条件黎曼算法、伪代码、收敛和复杂度", "5–7"),
                  ("5 Experiments", "OSSE、模式、真实资料、基线和消融", "7–10"),
                  ("6 Discussion", "低秩有效范围、不确定度、局限", "2–3"),
                  ("7 Conclusion", "结论和推广", "1"),
              ], [1800, 5700, 1860], font_size=8.8)

    doc.add_heading("12.4 可能被拒稿的情形", level=2)
    add_bullet(doc, "只把风场称为张量，然后调用CP/Tucker分解，没有新的反演模型。", bullet_id)
    add_bullet(doc, "把张量补全出的缺测数据当作真实观测输入PyDDA，却不给不确定度。", bullet_id)
    add_bullet(doc, "只在一个真实中气旋案例上展示漂亮流线，没有真值误差。", bullet_id)
    add_bullet(doc, "只与PyDDA默认参数比较，没有调优后的强基线和消融。", bullet_id)
    add_bullet(doc, "训练和测试包含同一风暴相邻体扫，造成数据泄漏。", bullet_id)
    add_bullet(doc, "理论结论依赖随机缺测/RIP，但实验缺测是结构化雷达扇区，假设与应用脱节。", bullet_id)
    add_bullet(doc, "没有代码、配置、随机种子或数据获取说明，结果不能复现。", bullet_id)

    doc.add_heading("13 研究进度建议", level=1)
    add_table(doc,
              ["阶段", "时间", "主要任务", "里程碑"],
              [
                  ("I", "第1–2月", "文献、Git基线、算子接口、连续体扫数据整理", "H/H*与D/D*测试通过"),
                  ("II", "第3–4月", "低秩经验分析、解析OSSE、Tucker原型", "证明低秩假设适用范围"),
                  ("III", "第5–7月", "黎曼算法、预条件、收敛与复杂度", "主算法稳定并优于无预条件版本"),
                  ("IV", "第8–9月", "模式OSSE、大规模敏感性和消融", "主表和误差图固定"),
                  ("V", "第10–11月", "真实多雷达验证、中气旋对象分析", "封闭测试集完成"),
                  ("VI", "第12月", "论文写作、代码清理、补充实验", "投稿初稿与复现包"),
              ], [800, 1200, 4800, 2560], font_size=8.7)

    doc.add_heading("14 风险、降级路线和决策点", level=1)
    add_table(doc,
              ["风险", "早期信号", "处理策略"],
              [
                  ("真实风场不低秩", "奇异值衰减缓慢、涡旋截断严重", "转为局地块/风暴坐标、低秩背景+稀疏旋转或仅做低秩预条件"),
                  ("差几何区不可恢复", "留出误差和不确定度持续很高", "承认不可辨识并输出置信掩膜；不要强制补全"),
                  ("理论条件过强", "RIP类条件无法对应结构缺测", "改做切空间限制稳定性与局部误差界"),
                  ("真实数据太少", "只有单案例且无负样本", "加强模式OSSE，同时继续获取跨站点季节资料"),
                  ("算法过慢", "每个窗口耗时远超PyDDA", "矩阵无关算子、秩截断、预条件、GPU/并行；报告精度—成本前沿"),
                  ("跨学科解释不足", "数学正确但气象指标异常", "邀请雷达气象合作者，建立人工判识和物理检查清单"),
              ], [2300, 2700, 4360], font_size=8.5)

    doc.add_heading("15 最终交付物清单", level=1)
    deliverables = [
        "一套矩阵无关雷达观测算子及严格伴随/梯度测试。",
        "一个物理约束低秩Tucker张量反演模型。",
        "一个几何感知预条件黎曼优化器及收敛诊断。",
        "解析OSSE和模式OSSE数据生成器。",
        "跨案例真实雷达评估数据集和封闭测试划分。",
        "PyDDA、直接反演、矩阵低秩、CP/Tucker/TT等基线。",
        "主实验、消融、敏感性、不确定度和复杂度报告。",
        "可复现代码、环境、配置、运行日志和论文绘图脚本。",
        "一篇以数学模型与算法为主的英文论文。",
    ]
    for item in deliverables:
        add_bullet(doc, item, bullet_id)

    doc.add_heading("附录A 符号表", level=1)
    add_table(doc,
              ["符号", "含义"],
              [
                  ("U / 𝒰", "三维或四维时空风场张量，最后一模为u、v、w"),
                  ("Hᵣ, Hᵣ*", "第r部雷达径向投影算子及其伴随"),
                  ("PΩᵣ", "第r部雷达有效观测掩膜算子"),
                  ("Wᵣ", "由BCA、SNR、距离和QC构成的权重张量"),
                  ("Dρ", "离散非弹性质量连续算子"),
                  ("T", "时间差分或平流残差算子"),
                  ("M_r", "给定多线性秩的Tucker张量流形"),
                  ("T_U M_r", "M_r在U处的切空间"),
                  ("σmin / κ", "最小奇异值 / 条件数，用于刻画可辨识性"),
                  ("R_U", "从切空间返回流形的retraction"),
              ], [2200, 7160], font_size=9)

    doc.add_heading("附录B 主算法伪代码", level=1)
    pseudo = [
        "输入：多雷达观测Y、掩膜Ω、几何信息、背景场U_b、多线性秩r、参数μ/γ。",
        "以U_b或PyDDA场做HOSVD，获得U₀∈M_r。",
        "对k=0,1,…：计算f(U_k)、观测残差、DρU_k和TU_k。",
        "调用H*、Dρ*、T*得到欧氏梯度，并投影得到grad f(U_k)。",
        "若‖grad f(U_k)‖/max(1,‖U_k‖)低于阈值，则停止。",
        "在T_{U_k}M_r中用预条件CG求解Gauss–Newton方向ξ_k。",
        "通过Armijo线搜索或信赖域选择步长α_k。",
        "令U_{k+1}=R_{U_k}(α_kξ_k)，必要时执行秩适应。",
        "输出：U、质量张量、目标历史、秩历史、停止原因和不确定度。",
    ]
    number_id = restart_numbering(doc, number_id)
    for line in pseudo:
        add_number(doc, line, number_id)

    doc.add_heading("参考文献", level=1)
    refs = [
        ("[1]", "Jackson, R. et al. PyDDA: A Pythonic Direct Data Assimilation Framework for Wind Retrievals. Journal of Open Research Software, 2020. DOI: 10.5334/jors.264."),
        ("[2]", "Gao, J., Xue, M., Shapiro, A., Droegemeier, K. A three-dimensional variational data analysis method with recursive filter for Doppler radars. Journal of Atmospheric and Oceanic Technology, 2004/related 3DVAR formulation."),
        ("[3]", "Shapiro, A., Potvin, C. K., Gao, J. Use of a Vertical Vorticity Equation in Variational Dual-Doppler Wind Analysis. JTECH, 2009. DOI: 10.1175/2009JTECHA1256.1."),
        ("[4]", "Potvin, C. K., Shapiro, A., Xue, M. Impact of a Vertical Vorticity Constraint in Variational Dual-Doppler Wind Analysis. JTECH, 2012. DOI: 10.1175/JTECH-D-11-00019.1."),
        ("[5]", "Stumpf, G. J. et al. The National Severe Storms Laboratory Mesocyclone Detection Algorithm for the WSR-88D. Weather and Forecasting, 1998. DOI: 10.1175/1520-0434(1998)013<0304:TNSSLM>2.0.CO;2."),
        ("[6]", "Kolda, T. G., Bader, B. W. Tensor Decompositions and Applications. SIAM Review, 2009. DOI: 10.1137/07070111X."),
        ("[7]", "Oseledets, I. V. Tensor-Train Decomposition. SIAM Journal on Scientific Computing, 2011. DOI: 10.1137/090752286."),
        ("[8]", "Zhang, Z., Aeron, S. Exact Tensor Completion Using t-SVD. IEEE Transactions on Signal Processing, 2017. DOI: 10.1109/TSP.2016.2639466."),
        ("[9]", "Kressner, D., Steinlechner, M., Vandereycken, B. Low-Rank Tensor Completion by Riemannian Optimization. BIT Numerical Mathematics, 2014. DOI: 10.1007/s10543-014-0475-z."),
        ("[10]", "Steinlechner, M. Riemannian Optimization for High-Dimensional Tensor Completion. SIAM Journal on Scientific Computing, 2016. DOI: 10.1137/15M1010506."),
        ("[11]", "Kressner, D., Steinlechner, M., Vandereycken, B. Preconditioned Low-Rank Riemannian Optimization for Linear Systems with Tensor Product Structure. SISC, 2016. DOI: 10.1137/15M1032909."),
        ("[12]", "Boyd, S. et al. Distributed Optimization and Statistical Learning via the Alternating Direction Method of Multipliers. Foundations and Trends in Machine Learning, 2011. DOI: 10.1561/2200000016."),
        ("[13]", "Yang, Y. et al. Tensor-Var: Efficient Four-Dimensional Variational Data Assimilation. ICML/PMLR, 2025."),
        ("[14]", "Collis, S. et al. Retrieval of Three-Dimensional Wind Fields from Doppler Radar Data Using an Efficient Two-Step Approach. Atmospheric Measurement Techniques, 2011, 4, 2717–2733."),
        ("[15]", "PyDDA User Guide: Overview and Retrieving Winds. Open Radar Science documentation, accessed 2026."),
    ]
    for label, citation in refs:
        p = doc.add_paragraph()
        p.paragraph_format.left_indent = Inches(0.35)
        p.paragraph_format.first_line_indent = Inches(-0.35)
        p.paragraph_format.space_after = Pt(5)
        r = p.add_run(label + " ")
        set_run_font(r, size=9, bold=True, color=DARK_BLUE)
        r = p.add_run(citation)
        set_run_font(r, size=9)

    doc.add_heading("网络资源", level=2)
    links = [
        ("PyDDA官方文档", "https://openradarscience.org/PyDDA/user_guide/overview.html"),
        ("SIAM Journal on Scientific Computing范围", "https://www.siam.org/publications/siam-journals/siam-journal-on-scientific-computing/"),
        ("Inverse Problems期刊范围", "https://publishingsupport.iopscience.iop.org/journals/inverse-problems/about-inverse-problems/"),
        ("Journal of Computational Physics范围", "https://doi.org/10.1006/jcph"),
        ("Journal of Atmospheric and Oceanic Technology范围", "https://www.ametsoc.org/ams/publications/journals/journal-of-atmospheric-and-oceanic-technology/"),
        ("Atmospheric Measurement Techniques范围", "https://www.atmospheric-measurement-techniques.net/"),
    ]
    for text, url in links:
        p = doc.add_paragraph()
        apply_num(p, bullet_id)
        add_hyperlink(p, text, url)

    doc.add_heading("结语", level=1)
    add_para(doc, "对于数学优化方向研究生，最有竞争力的工作不是证明“张量方法在一个雷达案例上更好”，而是回答三个普适问题：方向投影和结构缺测下何时可恢复低秩矢量场；几何病态如何进入稳定性常数和预条件器；如何在保证物理一致性的同时保留局地强旋转。现有项目已经提供数据入口、气象基线和失败案例，下一步应把主要精力投入观测算子、低秩流形算法、理论证明和可复现实验。")

    doc.core_properties.title = "交叉波束几何感知的物理约束低秩张量多普勒风场反演研究方案"
    doc.core_properties.subject = "硕士论文算法、模型、数学知识、数据、实验与项目改造方案"
    doc.core_properties.author = ""
    doc.core_properties.keywords = "tensor; Riemannian optimization; Doppler radar; wind retrieval; inverse problems"
    doc.core_properties.comments = "Generated as a research planning document."
    doc.save(DOCX_PATH)
    print(DOCX_PATH)


if __name__ == "__main__":
    build_document()
