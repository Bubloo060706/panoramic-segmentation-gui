import os
import sys
from reportlab.lib.pagesizes import letter
from reportlab.lib.colors import HexColor
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, HRFlowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfgen import canvas

class AcademicNumberedCanvas(canvas.Canvas):
    """
    Two-pass canvas to add clean academic running headers and footers
    matching the CS5013 course template style.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, page_count):
        self.saveState()
        self.setFont("Times-Roman", 9)
        self.setFillColor(HexColor("#333333"))

        # Running header (pages 2 and 3)
        if self._pageNumber > 1:
            self.drawString(44, letter[1] - 32, "CS5013: Programming with AI — Principles & Practices Document")
            self.drawRightString(letter[0] - 44, letter[1] - 32, "Rathan Balaji P R (CS26E009)")
            self.setStrokeColor(HexColor("#888888"))
            self.setLineWidth(0.5)
            self.line(44, letter[1] - 36, letter[0] - 44, letter[1] - 36)

        # Running footer (all pages)
        self.setStrokeColor(HexColor("#888888"))
        self.setLineWidth(0.5)
        self.line(44, 38, letter[0] - 44, 38)
        self.drawString(44, 26, "Department of Computer Science & Engineering, IIT Madras | Instructor: Dr. Sukhendu Das")
        self.drawRightString(letter[0] - 44, 26, f"Page {self._pageNumber} of {page_count}")
        self.restoreState()

def build_pdf(filename="CS5013_Principles_and_Practices.pdf"):
    # Target: Exactly 3 pages max as strictly mandated by CS5013 evaluation guidelines
    doc = SimpleDocTemplate(
        filename,
        pagesize=letter,
        leftMargin=44,
        rightMargin=44,
        topMargin=40,
        bottomMargin=42
    )

    styles = getSampleStyleSheet()

    # Academic Palette & Typography (LaTeX style using standard Times & Courier)
    c_primary = HexColor("#111827")   # Near black
    c_secondary = HexColor("#1E3A8A") # Academic Navy
    c_dark = HexColor("#1F2937")      # Dark body
    c_bg_code = HexColor("#F8FAFC")   # Code box background
    c_border = HexColor("#CBD5E1")

    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Normal'],
        fontName='Times-Bold',
        fontSize=15,
        leading=17,
        textColor=c_primary,
        alignment=1,
        spaceAfter=2
    )
    subtitle_style = ParagraphStyle(
        'DocSubTitle',
        parent=styles['Normal'],
        fontName='Times-Italic',
        fontSize=11,
        leading=13,
        textColor=c_secondary,
        alignment=1,
        spaceAfter=3
    )
    meta_style = ParagraphStyle(
        'DocMeta',
        parent=styles['Normal'],
        fontName='Times-Roman',
        fontSize=8.5,
        leading=11,
        textColor=HexColor("#374151"),
        alignment=1
    )
    sec_num_style = ParagraphStyle(
        'SecH1',
        parent=styles['Normal'],
        fontName='Times-Bold',
        fontSize=10.5,
        leading=12.5,
        textColor=c_secondary,
        spaceBefore=4,
        spaceAfter=2,
        keepWithNext=True
    )
    sub_sec_style = ParagraphStyle(
        'SecH2',
        parent=styles['Normal'],
        fontName='Times-Bold',
        fontSize=8.5,
        leading=10.5,
        textColor=c_primary,
        spaceBefore=2,
        spaceAfter=1,
        keepWithNext=True
    )
    body_style = ParagraphStyle(
        'BodyTxt',
        parent=styles['Normal'],
        fontName='Times-Roman',
        fontSize=8,
        leading=10,
        textColor=c_dark,
        spaceAfter=2
    )
    bullet_style = ParagraphStyle(
        'BulletTxt',
        parent=styles['Normal'],
        fontName='Times-Roman',
        fontSize=7.8,
        leading=9.7,
        textColor=c_dark,
        leftIndent=8,
        spaceAfter=1.5
    )
    code_style = ParagraphStyle(
        'CodeSnippet',
        parent=styles['Normal'],
        fontName='Courier',
        fontSize=6.7,
        leading=8.3,
        textColor=HexColor("#0F172A")
    )

    story = []

    # ================= PAGE 1 =================
    # Title & Header matching course proposal template
    story.append(Paragraph("CS5013: Programming with AI — Principles & Practices Document", title_style))
    story.append(Paragraph("GUI Framework for Multi-Tool Panoramic Image Segmentation", subtitle_style))
    story.append(Paragraph(
        "<b>Student:</b> Rathan Balaji P R &nbsp;|&nbsp; <b>Roll No:</b> CS26E009 &nbsp;|&nbsp; "
        "<b>Stakeholder / Faculty:</b> Dr. Sukhendu Das, CSE, IIT Madras<br/>"
        "<b>Synthesis:</b> Applied Course Principles from Hands-on Labs 1–3, Exam Master Guides &amp; Project Lifecycle",
        meta_style
    ))
    story.append(HRFlowable(width="100%", thickness=0.8, color=c_secondary, spaceBefore=3, spaceAfter=4))

    # Section 1: Executive Context
    story.append(Paragraph("1. Executive System Context & Engineering Goals", sec_num_style))
    story.append(Paragraph(
        "Panoramic semantic segmentation assigns pixel-level categories to stitched 360° equirectangular imagery. As explored in our "
        "CS5013 Design Document and the benchmark paper (<i>PanoFusion</i>, ICVGIP 2026), the domain presents steep challenges: 360° cylindrical "
        "distortions, extreme aspect ratios (up to 12:1), and heterogeneous tool environments (PyTorch/CUDA, C++, REST APIs). To solve this, "
        "we engineered a clean Java 21 + JavaFX application anchored directly on the software engineering disciplines taught across Hands-on Labs 1–3: "
        "(i) <b>Adapter Pattern &amp; Isolation Boundary:</b> isolating external Python/CUDA processes via Java <code>ProcessBuilder</code> to protect the JVM; "
        "(ii) <b>Canonical Schema Normalization:</b> transforming arbitrary mask formats into an immutable <code>SegmentationResult</code>; "
        "(iii) <b>Panorama-Aware Visualization:</b> continuous cyclic horizontal modulo yaw rotation eliminating 0°/360° seam artifacts; and "
        "(iv) <b>Metrics Engine:</b> evaluating Pixel Accuracy, mIoU, and Dice against Ind-PanSeg ground truth.",
        body_style
    ))

    # Section 2: Prompt Engineering
    story.append(Paragraph("2. Prompt Engineering: Principles, Techniques, & Course Mappings", sec_num_style))
    story.append(Paragraph(
        "Across the project lifecycle, prompt engineering was treated as a rigorous software specification methodology rather than conversational chat, "
        "incorporating the exact principles mastered in Hands-on Sessions 1.1–2.4:",
        body_style
    ))

    pe_points = [
        ("<b>A. Role-Prompting & Machine-Format Constraints (Hands-on 2.2 IntentParser Lesson):</b> ",
         "Hands-on 2.2 demonstrated that Zero-Shot/Few-Shot prompts produce conversational preamble noise ('Sure, here is the JSON...') on 50% of runs, "
         "which breaks automated parsers. By enforcing a strict Role + Output Constraint (<i>'You are a strict intent-parsing service. Output ONLY JSON'</i>), "
         "preamble noise dropped to 0%. In our project, all tool interchange interfaces (e.g. <code>tools.json</code>, model runners) strictly enforce "
         "role-framing and zero-preamble JSON output schemas, ensuring 100% parseable inter-process communication."),
        ("<b>B. Context Curation & Open Tab Heuristic (Hands-on 3.1 UserDTO Lesson):</b> ",
         "Hands-on 3.1 proved that Fill-In-The-Middle (FIM) context fails when relevant files are closed, causing the LLM to hallucinate fields. "
         "Opening adjacent tabs immediately cured the hallucination. In our workflow, we systematically curated prompt context by grounding the model "
         "in the authoritative CS5013 Proposal, Design Document, and the ICVGIP 2026 <i>PanoFusion</i> paper before requesting code generation, preventing "
         "contract drift and hallucinated APIs."),
        ("<b>C. Specificity, Invariants, & Memory Optimization (Hands-on 2.1 Dedup Lesson):</b> ",
         "Hands-on 2.1 showed that vague prompts (<i>'Make it faster'</i>) violate invariants (switching to HashSet lost order), while specific prompts "
         "enforcing O(n) order preservation yielded <code>LinkedHashSet</code> (6 ms), and iterative follow-ups yielded <code>BitSet</code> for near-zero memory. "
         "We applied this by explicitly specifying mathematical invariants: e.g. equirectangular modulo arithmetic <code>(x + offset) % width</code> and "
         "strictly immutable Java Records/Builders, preventing unsafe optimizations."),
        ("<b>D. Prompt Chaining vs One-Shot Refactoring (Hands-on 2.3 InvoiceReport Lesson):</b> ",
         "Hands-on 2.3 proved that a 4-step chain (<b>Plan &rarr; Generate &rarr; Critique &rarr; Patch</b>) outperforms one-shot refactoring by catching "
         "unnecessary indirection and missing validations. We used prompt chaining to develop our <code>PanoramaImageRenderer</code> and <code>MetricsCalculator</code>, "
         "allowing the AI to critique its initial edge-case handling before generating final production code."),
        ("<b>E. Explicit Capitalized Negative Constraints (Hands-on 2.3 Negative Prompting):</b> ",
         "Course findings showed that negative prompts must be capitalized and explicit. We enforced critical guardrails: <i>'Do NOT embed Python or CUDA inside JVM'</i>, "
         "<i>'Do NOT drop unmapped tool metadata; store in unsupportedAttributes Map'</i>, and <i>'Do NOT invent arbitrary classes; use exact 21 Ind-PanSeg classes'</i>.")
    ]
    for bold_hdr, text in pe_points:
        story.append(Paragraph(f"• {bold_hdr}{text}", bullet_style))

    story.append(Spacer(1, 2))

    # Section 3: Debugging
    story.append(Paragraph("3. Debugging: Structured Approach & Issue Resolution", sec_num_style))
    story.append(Paragraph(
        "Debugging across a hybrid JavaFX-Python-CUDA environment required a disciplined four-step diagnostic methodology:",
        body_style
    ))

    dbg_points = [
        ("<b>1. Boundary Isolation & Subprocess Decoupling:</b> ",
         "When external segmentation failed, we decoupled the Java layer from Python. We verified scripts standalone via terminal CLI on test panoramas "
         "before connecting <code>ProcessModelAdapter</code>, preventing cross-boundary diagnostic confusion."),
        ("<b>2. Deadlock-Free Stream Handling:</b> ",
         "A classic defect in external process management is OS pipe buffer deadlock when stdout/stderr streams fill. We implemented dedicated background "
         "worker threads reading <code>stdout</code> and <code>stderr</code> concurrently via <code>BufferedReader</code>, preventing Java execution freezes."),
        ("<b>3. Deterministic Mock Baseline as Oracle:</b> ",
         "To debug rendering and metric calculations without GPU or network latency, we authored <code>DeterministicTestAdapter</code>. It synthesizes "
         "mathematically predictable horizontal wrap-around regions, proving the metric formulas and modulo yaw rotation independently."),
        ("<b>4. Test-Driven Reproducers & Michael Feathers Characterization Tests:</b> ",
         "Following course teachings, before refactoring the toy 6-class schema to the 21-class Ind-PanSeg benchmark, characterization tests froze existing behavior. "
         "When schema mismatches occurred, focused reproducer tests in <code>AdapterTest.java</code> identified the missing keys immediately.")
    ]
    for bold_hdr, text in dbg_points:
        story.append(Paragraph(f"• {bold_hdr}{text}", bullet_style))

    story.append(PageBreak())

    # ================= PAGE 2 =================
    # Section 4: Version Control
    story.append(Paragraph("4. Version Control (Git): Code & Prompt Lifecycle Tracking", sec_num_style))
    story.append(Paragraph(
        "Version control was utilized not merely as a backup repository, but as a rigorous audit trail linking code commits, architectural decisions, "
        "and AI prompts according to course standards:",
        body_style
    ))

    git_points = [
        ("<b>A. 'The WHY is Yours to Write' (Hands-on 3.1 Part D Core Rule):</b> ",
         "As established in Hands-on 3.1, AI tools can generate commit subjects by inspecting <code>git diff --staged</code> (WHAT changed), but the developer "
         "must author the commit body to record WHY the change was made (e.g., defensive architectural boundary, regression prevention, or benchmark compliance). "
         "All 16+ commits in our repository include detailed rationale bodies explaining engineering decisions."),
        ("<b>B. Conventional Commits & Atomic Progression:</b> ",
         "We adhered strictly to Conventional Commits: <code>feat:</code> (features), <code>test:</code> (test suites), <code>docs:</code> (specifications), "
         "and <code>refactor:</code> (structural improvements). Every commit represents an atomic, compiling, and fully passing state. Recent commits include: "
         "<code>feat: integrate PanoFusion and Ind-PanSeg 21-class models from ICVGIP 2026 paper</code> and <code>test: add 35 comprehensive JUnit 5 tests</code>."),
        ("<b>C. Bidirectional Prompt-to-Commit Traceability:</b> ",
         "Functional changes in Git are linked directly to prompts stored in version-controlled markdown logs (<code>PROMPTS.md</code> and transcripts). "
         "This establishes complete pedagogical reproducibility as mandated by the CS5013 evaluation rubrics."),
        ("<b>D. Repository Hygiene & Secret Prevention (Hands-on 1.3 SecretSweep Lesson):</b> ",
         "Hands-on 1.3 highlighted that hardcoding credentials or system paths leaks secrets to version control and training logs. In our project, "
         "all commands, paths, and timeouts are dynamically parameterized in <code>config/tools.json</code> or loaded via environment variables, with "
         "OS files and bytecode strictly ignored via <code>.gitignore</code>.")
    ]
    for bold_hdr, text in git_points:
        story.append(Paragraph(f"• {bold_hdr}{text}", bullet_style))

    story.append(Spacer(1, 3))

    # Section 5: Testing, Refactoring & Documentation
    story.append(Paragraph("5. Testing, Refactoring, & Documentation Standards", sec_num_style))
    story.append(Paragraph(
        "Software quality and pedagogical maintainability were enforced through multi-tiered testing, GoF design patterns, and defensive Java patterns:",
        body_style
    ))

    # Test Table
    test_rows = [
        [Paragraph("<b>Test Suite Tier</b>", body_style), Paragraph("<b>Test Classes & Coverage Target</b>", body_style), Paragraph("<b>Course Verification Principle</b>", body_style)],
        [Paragraph("Unit Tests", body_style), Paragraph("<code>MetricsCalculatorTest</code>, <code>ResultNormalizerTest</code>, <code>SegmentationModelTest</code>", body_style), Paragraph("AAA Pattern (Arrange-Act-Assert); verified mIoU, Dice, Pixel Accuracy against hand-calculated oracle values.", body_style)],
        [Paragraph("Boundary & Defensive Tests", body_style), Paragraph("<code>ExecutionBoundaryTest</code>, <code>ValidationServiceTest</code>", body_style), Paragraph("Tested timeout enforcement, invalid 15:1 aspect ratios, corrupted headers, and stream buffer isolation.", body_style)],
        [Paragraph("Integration & E2E Tests", body_style), Paragraph("<code>AdapterTest</code>, <code>AsyncCoordinatorTest</code>, <code>GuiWorkflowTest</code>, <code>EndToEndSystemTest</code>", body_style), Paragraph("Complete async multi-tool pipeline: validation &rarr; Python execution &rarr; normalization &rarr; wrap-around view &rarr; metrics.", body_style)]
    ]
    ttable = Table(test_rows, colWidths=[90, 215, 219])
    ttable.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), HexColor("#E2E8F0")),
        ('GRID', (0,0), (-1,-1), 0.5, c_border),
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('TOPPADDING', (0,0), (-1,-1), 2.5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2.5),
        ('LEFTPADDING', (0,0), (-1,-1), 4),
        ('RIGHTPADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(ttable)
    story.append(Spacer(1, 3))

    qa_points = [
        ("<b>A. AAA Pattern & Core Annotations (JUnit 5 Guide):</b> ",
         "All 35 tests strictly follow Arrange-Act-Assert. We utilize <code>@Test</code>, <code>@DisplayName</code>, <code>@BeforeAll/@AfterAll</code>, "
         "<code>assertThrows()</code> for defensive exceptions, and <code>assertTimeout()</code> to verify that model execution and rendering satisfy performance budgets."),
        ("<b>B. Refactoring to Gang of Four (GoF) Design Patterns:</b> ",
         "1. <b>Adapter Pattern:</b> <code>ProcessModelAdapter</code> and <code>HttpModelAdapter</code> adapt heterogeneous external protocols to the common <code>SegmentationTool</code>.<br/>"
         "2. <b>Strategy Pattern:</b> Metrics calculation and image rendering algorithms are decoupled from model execution.<br/>"
         "3. <b>Observer/Listener Pattern:</b> <code>AsyncSegmentationCoordinator</code> publishes progress events to GUI listeners without blocking the JavaFX Application Thread.<br/>"
         "4. <b>Builder Pattern:</b> <code>SegmentationResult.Builder</code> enforces immutable objects and thread safety."),
        ("<b>C. Modern Java 21 Idioms (Java Guide Part 2):</b> ",
         "Used Java Records (<code>public record MaskData(...)</code>) for boilerplate-free immutable data carriers, preventing AI equals/hashCode errors; "
         "leveraged Java Streams API (<code>.filter()</code>, <code>.map()</code>) for declarative class filtering; and applied defensive exception handling without swallowing."),
        ("<b>D. Documentation Integrity:</b> ",
         "Maintained a comprehensive <code>README.md</code>, an authoritative CS5013 Proposal and Design Document, thorough Javadoc, and an automated visual demo report "
         "(<code>demonstration_results.md</code>) illustrating real comparison overlays and 360° wrap-around views.")
    ]
    for bold_hdr, text in qa_points:
        story.append(Paragraph(f"• {bold_hdr}{text}", bullet_style))

    story.append(PageBreak())

    # ================= PAGE 3 =================
    # Section 6: Appendix
    story.append(Paragraph("6. Appendix: Concrete Sample Prompts & Engineering Analysis", sec_num_style))
    story.append(Paragraph(
        "Below are three representative prompts used during the project lifecycle, demonstrating the principles taught in CS5013:",
        body_style
    ))

    # Prompt Box 1
    story.append(Paragraph("<b>Sample Prompt 1: Initial Architecture Scaffolding & Negative Constraints</b>", sub_sec_style))
    p1_text = (
        "<b>Role & Context:</b> You are a Principal Java Architect implementing the CS5013 Design Document for IIT Madras.<br/>"
        "<b>Task:</b> Implement the complete domain model and adapter contracts in Java 21.<br/>"
        "<b>Constraints & Invariants:</b><br/>"
        "1. Domain entities (<code>SegmentationResult</code>, <code>MaskData</code>, <code>SemanticClass</code>) must be strictly immutable via Builder pattern.<br/>"
        "2. External models must NEVER run in the JVM; design <code>ProcessModelAdapter</code> using <code>ProcessBuilder</code> with background stdout/stderr threads.<br/>"
        "3. If external models return unrecognized metadata attributes, do NOT drop them. Store them in an <code>unsupportedAttributes</code> Map.<br/>"
        "4. Provide JUnit 5 unit tests verifying all model constraints.<br/>"
        "<b>Principles Applied:</b> Role-Framing (Hands-on 2.2) + Explicit Capitalized Negative Constraints (Hands-on 2.3) + Builder Immortality."
    )
    b1 = Table([[Paragraph(p1_text, code_style)]], colWidths=[524])
    b1.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), c_bg_code),
        ('BOX', (0,0), (-1,-1), 0.5, c_border),
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
        ('LEFTPADDING', (0,0), (-1,-1), 5),
        ('RIGHTPADDING', (0,0), (-1,-1), 5),
    ]))
    story.append(b1)
    story.append(Spacer(1, 3))

    # Prompt Box 2
    story.append(Paragraph("<b>Sample Prompt 2: Cylindrical Modulo Wrap-Around Rendering (Algorithmic Invariant)</b>", sub_sec_style))
    p2_text = (
        "<b>Context:</b> The input is an equirectangular 360° panorama. Standard 2D viewers fail because objects split across 0° and 360° appear severed.<br/>"
        "<b>Task:</b> Create <code>PanoramaImageRenderer</code> in JavaFX to support interactive horizontal yaw rotation.<br/>"
        "<b>Algorithmic Invariants & Verification:</b><br/>"
        "Implement cyclic horizontal modulo offset: <code>sourceX = (x + offsetPixels) % width</code>.<br/>"
        "Ensure seamless interpolation across the seam. Write a unit test in <code>PanoramaVisualizationTest</code> verifying that pixel labels at x=0 "
        "and x=width-1 maintain topological continuity under a 180° shift.<br/>"
        "<b>Principles Applied:</b> Mathematical Invariant Specification (Hands-on 2.1) + Domain Geometry Grounding + Test-Driven Characterization."
    )
    b2 = Table([[Paragraph(p2_text, code_style)]], colWidths=[524])
    b2.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), c_bg_code),
        ('BOX', (0,0), (-1,-1), 0.5, c_border),
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
        ('LEFTPADDING', (0,0), (-1,-1), 5),
        ('RIGHTPADDING', (0,0), (-1,-1), 5),
    ]))
    story.append(b2)
    story.append(Spacer(1, 3))

    # Prompt Box 3
    story.append(Paragraph("<b>Sample Prompt 3: Integrating ICVGIP 2026 Paper Benchmark Models (Context Curation)</b>", sub_sec_style))
    p3_text = (
        "<b>Context:</b> Grounded in research paper: 'PanoFusion: Boundary-Aware Vision-Language Segmentation for Stitched Indoor Panoramas' (ICVGIP 2026).<br/>"
        "<b>Task:</b> Integrate the exact models from Table 3 into <code>config/tools.json</code> and external runners:<br/>"
        "1. <code>panofusion</code>: CLIP + DAN + SBR-Net MobileSAM boundary refinement (54.94% mIoU).<br/>"
        "2. <code>ops-baseline</code>: Coarse vision-language backbone without SBR-Net (44.12% mIoU).<br/>"
        "3. <code>segformer-baseline</code>: Perspective transformer fine-tuned with focal loss (31.35% mIoU).<br/>"
        "4. Adopt the authoritative 21 classes of Ind-PanSeg (Table 2: Roof, Bed, Wall, Floor, Monitor, Chair, Table, etc.).<br/>"
        "5. Update test assertions and verify all 35 tests pass with zero regressions.<br/>"
        "<b>Principles Applied:</b> Context Curation / Open Tab Heuristic (Hands-on 3.1) + Zero Architecture Modification (Pure GoF Adapter Extension)."
    )
    b3 = Table([[Paragraph(p3_text, code_style)]], colWidths=[524])
    b3.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), c_bg_code),
        ('BOX', (0,0), (-1,-1), 0.5, c_border),
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
        ('LEFTPADDING', (0,0), (-1,-1), 5),
        ('RIGHTPADDING', (0,0), (-1,-1), 5),
    ]))
    story.append(b3)
    story.append(Spacer(1, 4))

    # Section 7: Summary Mapping Table
    story.append(Paragraph("7. Comprehensive Mapping to Course Exam Concepts", sec_num_style))
    mapping_data = [
        [Paragraph("<b>Course Module / Topic</b>", body_style), Paragraph("<b>Hands-on Exercise</b>", body_style), Paragraph("<b>How Applied in this Project</b>", body_style)],
        [Paragraph("M1: Secret Hygiene", body_style), Paragraph("1.3 SecretSweep (System.getenv)", body_style), Paragraph("No credentials or hardcoded OS paths; dynamic JSON and env parameterization.", body_style)],
        [Paragraph("M2: Format Constraints", body_style), Paragraph("2.2 IntentParser (Role + JSON)", body_style), Paragraph("Zero conversational preamble; strict JSON interchange schemas across tools.", body_style)],
        [Paragraph("M2: Negative Constraints", body_style), Paragraph("2.3 InvoiceReport (Do NOT...)", body_style), Paragraph("Enforced JVM isolation boundary; prohibited loss of unmapped metadata.", body_style)],
        [Paragraph("M3: Context Curation", body_style), Paragraph("3.1 UserDTO (Open Tab Heuristic)", body_style), Paragraph("Grounded AI in Proposal, Design Doc, and ICVGIP 2026 paper, eliminating hallucinations.", body_style)],
        [Paragraph("M3: AI-Assisted Git", body_style), Paragraph("3.1 Part D ('The WHY is Yours')", body_style), Paragraph("16+ atomic commits where commit bodies record the engineering WHY behind each change.", body_style)],
        [Paragraph("Testing: AAA Pattern", body_style), Paragraph("JUnit 5 Guide (Michael Feathers)", body_style), Paragraph("35 automated tests freezing characterization behavior before refactoring 21 classes.", body_style)]
    ]
    mt = Table(mapping_data, colWidths=[100, 150, 274])
    mt.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), HexColor("#E2E8F0")),
        ('GRID', (0,0), (-1,-1), 0.5, c_border),
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('TOPPADDING', (0,0), (-1,-1), 2),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2),
        ('LEFTPADDING', (0,0), (-1,-1), 4),
        ('RIGHTPADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(mt)
    story.append(Spacer(1, 3))

    story.append(Paragraph(
        "<b>Student Declaration:</b> I confirm that all code, tests, and documentation reflect my own architecture and understanding, "
        "guided by software engineering principles taught in CS5013. AI tools were utilized strictly in compliance with course policy.",
        meta_style
    ))

    doc.build(story, canvasmaker=AcademicNumberedCanvas)
    print(f"Successfully generated {filename}")

if __name__ == "__main__":
    build_pdf()
