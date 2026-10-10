import os
import sys
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib.colors import HexColor
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether, HRFlowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfgen import canvas

class NumberedCanvas(canvas.Canvas):
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
        self.setFont("Helvetica", 8)
        self.setFillColor(HexColor("#555555"))
        # Header (pages 2+)
        if self._pageNumber > 1:
            self.drawString(54, letter[1] - 36, "CS5013: Programming with AI — Engineering Principles & Practices Document")
            self.drawRightString(letter[0] - 54, letter[1] - 36, "Roll No: CS26E009")
            self.setStrokeColor(HexColor("#CCCCCC"))
            self.setLineWidth(0.5)
            self.line(54, letter[1] - 42, letter[0] - 54, letter[1] - 42)

        # Footer (all pages)
        self.setStrokeColor(HexColor("#CCCCCC"))
        self.setLineWidth(0.5)
        self.line(54, 45, letter[0] - 54, 45)
        self.drawString(54, 32, "IIT Madras | Dept. of Computer Science & Engineering | Dr. Sukhendu Das")
        self.drawRightString(letter[0] - 54, 32, f"Page {self._pageNumber} of {page_count}")
        self.restoreState()

def build_pdf(filename="CS5013_Principles_and_Practices.pdf"):
    # Margins: 0.55 in (approx 40 pt) to fit rich, tight content cleanly in 3 pages
    doc = SimpleDocTemplate(
        filename,
        pagesize=letter,
        leftMargin=44,
        rightMargin=44,
        topMargin=46,
        bottomMargin=46
    )

    styles = getSampleStyleSheet()

    # Custom typography
    c_primary = HexColor("#1A365D")   # Deep navy
    c_secondary = HexColor("#2B6CB0") # Medium blue
    c_dark = HexColor("#2D3748")      # Charcoal body
    c_bg_code = HexColor("#F7FAFC")   # Light gray
    c_border = HexColor("#CBD5E0")

    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=15,
        leading=18,
        textColor=c_primary,
        alignment=1
    )
    meta_style = ParagraphStyle(
        'DocMeta',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8.5,
        leading=11.5,
        textColor=HexColor("#4A5568"),
        alignment=1
    )
    h1_style = ParagraphStyle(
        'SecH1',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=11,
        leading=13.5,
        textColor=c_primary,
        spaceBefore=5,
        spaceAfter=3,
        keepWithNext=True
    )
    h2_style = ParagraphStyle(
        'SecH2',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=9,
        leading=11.5,
        textColor=c_secondary,
        spaceBefore=3,
        spaceAfter=2,
        keepWithNext=True
    )
    body_style = ParagraphStyle(
        'BodyTxt',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8,
        leading=10.5,
        textColor=c_dark,
        spaceAfter=3
    )
    bullet_style = ParagraphStyle(
        'BulletTxt',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=7.8,
        leading=10.2,
        textColor=c_dark,
        leftIndent=10,
        spaceAfter=2
    )
    code_style = ParagraphStyle(
        'CodeSnippet',
        parent=styles['Normal'],
        fontName='Courier',
        fontSize=7,
        leading=8.8,
        textColor=HexColor("#1A202C")
    )

    story = []

    # ================= PAGE 1 =================
    # Title & Metadata
    story.append(Paragraph("CS5013: Programming with AI — Principles & Practices Document", title_style))
    story.append(Spacer(1, 3))
    story.append(Paragraph(
        "<b>Project:</b> GUI Framework for Multi-Tool Panoramic Image Segmentation &nbsp;|&nbsp; "
        "<b>Student:</b> Rathan Balaji P R (CS26E009)<br/>"
        "<b>Stakeholder / Instructor:</b> Dr. Sukhendu Das &nbsp;|&nbsp; Department of Computer Science & Engineering, IIT Madras",
        meta_style
    ))
    story.append(HRFlowable(width="100%", thickness=1, color=c_primary, spaceBefore=4, spaceAfter=5))

    # Section 0: Project Context & Architecture Overview
    story.append(Paragraph("1. Executive Summary & Architectural Overview", h1_style))
    story.append(Paragraph(
        "Panoramic image segmentation introduces severe geometric challenges: 360° cylindrical equirectangular distortion, "
        "aspect ratios up to 12:1, and heterogeneous tool runtimes (Python/PyTorch, REST APIs, C++). To unify these tools, "
        "we engineered a modular Java 21 + JavaFX desktop framework adhering to clean architecture. "
        "Core components include: (i) <b>Model Isolation Boundary:</b> heavy deep-learning dependencies execute in isolated external "
        "processes via Java <code>ProcessBuilder</code>; (ii) <b>Canonical Normalization:</b> outputs map into a uniform "
        "<code>SegmentationResult</code> schema without losing unmapped metadata; (iii) <b>Panorama-Aware View:</b> interactive 360° horizontal "
        "modulo yaw wrap-around; and (iv) <b>Metrics Engine:</b> evaluating Pixel Accuracy, mIoU, and Dice score against Ind-PanSeg ground truth.",
        body_style
    ))

    # Section 1: Prompt Engineering
    story.append(Paragraph("2. Prompt Engineering Principles & Lifecycle Application", h1_style))
    story.append(Paragraph(
        "Throughout this project, prompt engineering was treated as a rigorous software specification discipline rather than ad-hoc chat. "
        "We systematically applied five core principles:",
        body_style
    ))

    p_principles = [
        ("<b>A. Role-Framing & System Identity:</b> ",
         "Each interaction explicitly designated the AI as a Principal Software Architect paired with a Graduate Student at IIT Madras. "
         "This anchored the model to academic rigor, strict design pattern compliance (GoF Adapter, Strategy, Observer), and zero tolerance for stub code."),
        ("<b>B. Context Bounding & In-Context Grounding:</b> ",
         "Rather than allowing the model to hallucinate specifications, prompts provided authoritative technical anchors: the CS5013 Project Proposal, "
         "the Project Design Document, and the ICVGIP 2026 research paper (<i>PanoFusion</i>, Submission 214). File paths and exact method contracts were bounded."),
        ("<b>C. Negative Prompting & Anti-Pattern Constraints:</b> ",
         "Prompts strictly enforced boundaries: <i>'Do NOT embed Python or CUDA inside JVM'</i>, <i>'Do NOT silently drop unsupported tool metadata'</i>, "
         "<i>'Do NOT invent arbitrary classes; use exact Ind-PanSeg 21 classes'</i>, and <i>'Do NOT use third-party CSS; use clean JavaFX CSS'</i>."),
        ("<b>D. Output Structuring & Few-Shot Schemas:</b> ",
         "We enforced exact JSON schemas for tool interchange (e.g. <code>tools.json</code>) and strict POJO Builder contracts (<code>SegmentationResult.Builder</code>). "
         "This ensured bidirectional serialization without contract drift."),
        ("<b>E. Iterative Chain-of-Thought & Stepwise Verification:</b> ",
         "Complex implementations were broken into discrete verified increments: Domain Model &rarr; Adapters &rarr; Tests &rarr; UI Rendering. "
         "Each step was verified through test execution before progressing to the next layer.")
    ]
    for bold_hdr, text in p_principles:
        story.append(Paragraph(f"• {bold_hdr}{text}", bullet_style))

    story.append(Spacer(1, 4))
    story.append(Paragraph("3. Debugging: Structured Approach & Issue Resolution", h1_style))
    story.append(Paragraph(
        "Debugging across a hybrid JavaFX-Python-CUDA environment requires a rigorous scientific methodology to isolate runtime boundaries:",
        body_style
    ))

    d_steps = [
        ("<b>1. Boundary Isolation (JVM vs Subprocess):</b> ",
         "When model execution failed, we decoupled the Java adapter from the Python script. Python runners were verified standalone via CLI "
         "with known sample panoramas before connecting the Java <code>ProcessModelAdapter</code>, preventing cross-boundary diagnostic confusion."),
        ("<b>2. Deadlock-Free Stream Consumption:</b> ",
         "A critical bug in external process execution is OS pipe deadlock when standard output/error buffers fill. We implemented separate worker "
         "threads consuming <code>stdout</code> and <code>stderr</code> concurrently using <code>BufferedReader</code>, preventing Java execution freezes."),
        ("<b>3. Deterministic Mock Test Baselines:</b> ",
         "To debug rendering and evaluation logic without GPU/network latency, we created <code>DeterministicTestAdapter</code>. It synthesizes "
         "mathematically predictable horizontal wrap-around regions, proving the metric formulas and modulo yaw rotation independently."),
        ("<b>4. Test-Driven Reproducers:</b> ",
         "Whenever a bug emerged (e.g. schema mismatches when switching to 21-class Ind-PanSeg), we authored a focused failing test in "
         "<code>AdapterTest.java</code>, debugged the failure trace, updated the normalization parser, and re-ran tests until green.")
    ]
    for bold_hdr, text in d_steps:
        story.append(Paragraph(f"• {bold_hdr}{text}", bullet_style))

    story.append(PageBreak())

    # ================= PAGE 2 =================
    story.append(Paragraph("4. Version Control (Git): Code & Prompt Lifecycle Tracking", h1_style))
    story.append(Paragraph(
        "Version control was leveraged not merely as an archive, but as an auditable trail of architectural decisions, code evolutions, and AI prompts:",
        body_style
    ))

    git_points = [
        ("<b>A. Conventional Atomic Commits:</b> ",
         "We adhered strictly to the Conventional Commits specification (<code>feat:</code>, <code>test:</code>, <code>docs:</code>, <code>refactor:</code>). "
         "Every commit represents an atomic, compiling state with passing test suites. Examples: <code>feat: implement core domain model</code>, "
         "<code>test: add 35 comprehensive JUnit 5 tests</code>, <code>feat: integrate PanoFusion and Ind-PanSeg 21-class models from ICVGIP 2026 paper</code>."),
        ("<b>B. Bidirectional Prompt-to-Commit Tracking:</b> ",
         "Commit messages explicitly link functional changes to prompt specifications. In addition, the complete prompt-response trajectories are preserved "
         "in versioned Markdown artifacts and transcript logs, ensuring full reproducibility and auditability required by the course guidelines."),
        ("<b>C. Build Hygiene & Configuration Versioning:</b> ",
         "The Maven wrapper (<code>mvnw.cmd</code>), dependencies, and external tool definitions (<code>config/tools.json</code>) are strictly version-controlled. "
         "Local machine-specific artifacts (e.g. <code>.m2</code> cache, OS dumps, compiled bytecode) are isolated via an exhaustive <code>.gitignore</code>."),
        ("<b>D. Linear, Rebase-Friendly History:</b> ",
         "We maintain a clean linear Git history without redundant merge bubbles, allowing seamless rollback and crystal-clear delta reviews for TAs.")
    ]
    for bold_hdr, text in git_points:
        story.append(Paragraph(f"• {bold_hdr}{text}", bullet_style))

    story.append(Spacer(1, 4))
    story.append(Paragraph("5. Testing, Refactoring & Documentation Standards", h1_style))
    story.append(Paragraph(
        "High code quality and pedagogical maintainability were enforced through multi-tier testing, design pattern refactoring, and comprehensive documentation:",
        body_style
    ))

    # Table of Test Suites
    test_data = [
        [Paragraph("<b>Test Category</b>", body_style), Paragraph("<b>Test Classes & Target</b>", body_style), Paragraph("<b>Coverage & Verification Target</b>", body_style)],
        [Paragraph("Unit Tests", body_style), Paragraph("<code>MetricsCalculatorTest</code>, <code>ResultNormalizerTest</code>, <code>SegmentationModelTest</code>", body_style), Paragraph("Verified mathematical correctness of Pixel Accuracy, mIoU, Dice; JSON schema parser; boundary wrap logic.", body_style)],
        [Paragraph("Boundary & Execution Tests", body_style), Paragraph("<code>ExecutionBoundaryTest</code>, <code>ValidationServiceTest</code>", body_style), Paragraph("Tested timeout enforcement, invalid aspect ratio handling, missing tools, and stream buffer isolation.", body_style)],
        [Paragraph("Integration & UI Tests", body_style), Paragraph("<code>AdapterTest</code>, <code>AsyncCoordinatorTest</code>, <code>GuiWorkflowTest</code>, <code>EndToEndSystemTest</code>", body_style), Paragraph("Complete asynchronous multi-tool pipeline: image validation &rarr; Python execution &rarr; normalization &rarr; wrap-around view &rarr; metrics.", body_style)]
    ]
    t = Table(test_data, colWidths=[90, 210, 224])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), HexColor("#E2E8F0")),
        ('GRID', (0,0), (-1,-1), 0.5, c_border),
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
        ('LEFTPADDING', (0,0), (-1,-1), 4),
        ('RIGHTPADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(t)
    story.append(Spacer(1, 4))

    refactor_points = [
        ("<b>Refactoring to GoF Design Patterns:</b> ",
         "1. <b>Adapter Pattern:</b> <code>ProcessModelAdapter</code> and <code>HttpModelAdapter</code> adapt heterogeneous external protocols to <code>SegmentationTool</code>.<br/>"
         "2. <b>Strategy Pattern:</b> Metrics calculation and image rendering algorithms are decoupled from model execution.<br/>"
         "3. <b>Observer/Listener Pattern:</b> <code>AsyncSegmentationCoordinator</code> publishes progress events to GUI listeners without blocking JavaFX Application Thread.<br/>"
         "4. <b>Builder Pattern:</b> <code>SegmentationResult.Builder</code> guarantees immutability and complete audit tracking."),
        ("<b>Documentation Integrity:</b> ",
         "The repository features a professional <code>README.md</code>, an authoritative CS5013 Proposal and Design Document, comprehensive Javadoc on all public API classes, "
         "and an automated demonstration report (<code>demonstration_results.md</code>) illustrating real visual comparison overlays and 360 wrap-around views.")
    ]
    for bold_hdr, text in refactor_points:
        story.append(Paragraph(f"• {bold_hdr}{text}", bullet_style))

    story.append(PageBreak())

    # ================= PAGE 3 =================
    story.append(Paragraph("6. Appendix: Concrete Sample Prompts & Prompt Engineering Analysis", h1_style))
    story.append(Paragraph(
        "Below are actual prompts employed across the development lifecycle, annotated with the engineering principles applied:",
        body_style
    ))

    # Box 1
    story.append(Paragraph("<b>Sample Prompt 1: Initial Architecture Scaffolding & Contract Enforcement</b>", h2_style))
    box1_content = [
        [Paragraph(
            "<b>Role & Context:</b> You are a Principal Java/Software Architect implementing the CS5013 Design Document for IIT Madras.<br/>"
            "<b>Task:</b> Implement the complete domain model and adapter contracts in Java 21.<br/>"
            "<b>Constraints:</b><br/>"
            "1. Model classes (<code>SegmentationResult</code>, <code>MaskData</code>, <code>SemanticClass</code>) must be strictly immutable using Builder pattern.<br/>"
            "2. External models must NEVER run in the JVM; design <code>ProcessModelAdapter</code> using <code>ProcessBuilder</code> with background stdout/stderr threads.<br/>"
            "3. If external models return unrecognized metadata attributes, do NOT drop them. Store them in an <code>unsupportedAttributes</code> Map.<br/>"
            "4. Provide JUnit 5 unit tests verifying all model constraints.<br/>"
            "<b>Engineering Principle Demonstrated:</b> Role-framing + Negative Prompting (preventing JVM pollution) + Concrete Contract Schemas.",
            code_style
        )]
    ]
    t1 = Table(box1_content, colWidths=[524])
    t1.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), c_bg_code),
        ('BOX', (0,0), (-1,-1), 0.5, c_border),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ('LEFTPADDING', (0,0), (-1,-1), 6),
        ('RIGHTPADDING', (0,0), (-1,-1), 6),
    ]))
    story.append(t1)
    story.append(Spacer(1, 4))

    # Box 2
    story.append(Paragraph("<b>Sample Prompt 2: Cylindrical Wrap-Around Rendering & Boundary Continuity</b>", h2_style))
    box2_content = [
        [Paragraph(
            "<b>Context:</b> The input is an equirectangular 360° panorama. Standard 2D viewers fail because objects split across 0° and 360° appear severed.<br/>"
            "<b>Task:</b> Create <code>PanoramaImageRenderer</code> in JavaFX to support interactive horizontal yaw rotation.<br/>"
            "<b>Algorithm Requirement:</b><br/>"
            "Implement cyclic modulo offset: <code>sourceX = (x + offsetPixels) % width</code>.<br/>"
            "Ensure seamless interpolation across the seam. Write a unit test verifying that pixel labels at x=0 and x=width-1 retain geometric continuity.<br/>"
            "<b>Engineering Principle Demonstrated:</b> Algorithmic Precision + Domain Geometry Grounding + Test-Driven Specification.",
            code_style
        )]
    ]
    t2 = Table(box2_content, colWidths=[524])
    t2.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), c_bg_code),
        ('BOX', (0,0), (-1,-1), 0.5, c_border),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ('LEFTPADDING', (0,0), (-1,-1), 6),
        ('RIGHTPADDING', (0,0), (-1,-1), 6),
    ]))
    story.append(t2)
    story.append(Spacer(1, 4))

    # Box 3
    story.append(Paragraph("<b>Sample Prompt 3: Integrating ICVGIP 2026 Paper Benchmark Models</b>", h2_style))
    box3_content = [
        [Paragraph(
            "<b>Context:</b> Research paper: 'PanoFusion: Boundary-Aware Vision-Language Segmentation for Stitched Indoor Panoramas' (ICVGIP 2026).<br/>"
            "<b>Task:</b> Integrate the exact models from Table 3 into <code>config/tools.json</code> and external runners:<br/>"
            "1. <code>panofusion</code>: CLIP + DAN + SBR-Net MobileSAM boundary refinement (54.94% mIoU).<br/>"
            "2. <code>ops-baseline</code>: Coarse vision-language backbone without SBR-Net (44.12% mIoU).<br/>"
            "3. <code>segformer-baseline</code>: Perspective transformer fine-tuned with focal loss (31.35% mIoU).<br/>"
            "4. Adopt the authoritative 21 classes of Ind-PanSeg (Table 2: Roof, Bed, Wall, Floor, Monitor, Chair, Table, etc.).<br/>"
            "5. Update test assertions and verify all 35 tests pass with zero regressions.<br/>"
            "<b>Engineering Principle Demonstrated:</b> Zero Code Modification to Core Architecture (Pure Adapter Pattern Extension) + Real Benchmark Grounding.",
            code_style
        )]
    ]
    t3 = Table(box3_content, colWidths=[524])
    t3.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), c_bg_code),
        ('BOX', (0,0), (-1,-1), 0.5, c_border),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ('LEFTPADDING', (0,0), (-1,-1), 6),
        ('RIGHTPADDING', (0,0), (-1,-1), 6),
    ]))
    story.append(t3)
    story.append(Spacer(1, 5))

    # Concluding declaration
    story.append(Paragraph(
        "<b>Student Declaration:</b> I confirm that all code, tests, and documentation reflect my own architecture and understanding, "
        "guided by software engineering best practices taught in CS5013. AI tools were utilized strictly in compliance with course policy.",
        meta_style
    ))

    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"Successfully generated {filename}")

if __name__ == "__main__":
    build_pdf()
