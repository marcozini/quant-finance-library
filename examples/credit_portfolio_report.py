#### Credit Portfolio PDF Report ####

from io import BytesIO

import matplotlib.pyplot as plt
import numpy as np

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


# ---------------------------------------------------------
# PDF-specific plots
# ---------------------------------------------------------

# Create a PDF-specific exposure chart showing only the largest exposures.
def create_report_exposure_plot(
    portfolio,
    base_currency,
    top_n=8,
):

    plot_data = portfolio.nlargest(
        top_n,
        "exposure_base",
    ).sort_values(
        "exposure_base",
        ascending=False,
    )

    buffer = BytesIO()

    plt.figure(figsize=(10, 6))

    plt.bar(
        plot_data["counterparty"],
        plot_data["exposure_base"],
    )

    plt.title(
        f"Top {len(plot_data)} Counterparty Exposures"
    )

    plt.xlabel(
        "Counterparty"
    )

    plt.ylabel(
        f"Exposure ({base_currency})"
    )

    plt.xticks(
        rotation=45,
        ha="right",
    )

    plt.tight_layout()

    plt.savefig(
        buffer,
        format="png",
        dpi=200,
    )

    plt.close()

    buffer.seek(0)

    return buffer


# Create a PDF-specific risk contribution chart for largest exposures.
def create_report_risk_contribution_plot(
    portfolio,
    confidence,
    base_currency,
    top_n=8,
):

    # Use the same top exposures as the exposure plot.
    plot_data = portfolio.nlargest(
        top_n,
        "exposure_base",
    ).sort_values(
        "exposure_base",
        ascending=False,
    )

    positions = np.arange(
        len(plot_data)
    )

    width = 0.38

    buffer = BytesIO()

    plt.figure(figsize=(11, 6))

    plt.bar(
        positions - width / 2,
        plot_data["incremental_var"],
        width,
        label=f"Incremental {confidence:.1%} VaR",
    )

    plt.bar(
        positions + width / 2,
        plot_data["incremental_es"],
        width,
        label=f"Incremental {confidence:.1%} ES",
    )

    plt.title(
        f"Tail-Risk Contributions — Top {len(plot_data)} Exposures"
    )

    plt.xlabel(
        "Counterparty"
    )

    plt.ylabel(
        f"Risk contribution ({base_currency})"
    )

    plt.xticks(
        positions,
        plot_data["counterparty"],
        rotation=45,
        ha="right",
    )

    plt.legend()
    plt.tight_layout()

    plt.savefig(
        buffer,
        format="png",
        dpi=200,
    )

    plt.close()

    buffer.seek(0)

    return buffer


# ---------------------------------------------------------
# PDF report
# ---------------------------------------------------------

# Create the compact two-page credit portfolio PDF report.
def create_credit_portfolio_report(
    results,
    output_pdf,
    loss_distribution_plot,
    tail_distribution_plot,
):

    portfolio = results[
        "portfolio"
    ]

    settings = results[
        "settings"
    ]

    summary = results[
        "summary"
    ]

    base_currency = summary[
        "base_currency"
    ]

    confidence = summary[
        "confidence_level"
    ]

    total_exposure = portfolio[
        "exposure_base"
    ].sum()

    # Largest deterministic single-name default loss.
    default_losses = (
        portfolio["exposure_base"]
        * portfolio["lgd"]
    )

    largest_index = default_losses.idxmax()

    largest_default_loss = float(
        default_losses.loc[
            largest_index
        ]
    )

    largest_counterparty = portfolio.loc[
        largest_index,
        "counterparty",
    ]

    # ---------------------------------------------------------
    # Document setup
    # ---------------------------------------------------------

    document = SimpleDocTemplate(
        str(output_pdf),
        pagesize=landscape(A4),
        leftMargin=1.0 * cm,
        rightMargin=1.0 * cm,
        topMargin=0.8 * cm,
        bottomMargin=0.8 * cm,
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "ReportTitle",
        parent=styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=18,
        leading=21,
        alignment=TA_CENTER,
        textColor=colors.HexColor(
            "#1F4E78"
        ),
        spaceAfter=5,
    )

    subtitle_style = ParagraphStyle(
        "Subtitle",
        parent=styles["BodyText"],
        fontSize=8.5,
        leading=11,
        alignment=TA_CENTER,
        textColor=colors.HexColor(
            "#555555"
        ),
        spaceAfter=8,
    )

    section_style = ParagraphStyle(
        "Section",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=11,
        leading=13,
        textColor=colors.HexColor(
            "#1F4E78"
        ),
        spaceBefore=4,
        spaceAfter=5,
    )

    note_style = ParagraphStyle(
        "Note",
        parent=styles["BodyText"],
        fontSize=7.5,
        leading=9.5,
        textColor=colors.HexColor(
            "#444444"
        ),
    )

    table_text_style = ParagraphStyle(
        "TableText",
        parent=styles["BodyText"],
        fontSize=7,
        leading=8,
    )

    story = []

    # =========================================================
    # Page 1 - Portfolio Risk Overview
    # =========================================================

    story.append(
        Paragraph(
            "Credit Portfolio Risk Report",
            title_style,
        )
    )

    dependence_model = str(
        settings[
            "dependence_model"
        ]
    ).replace(
        "_",
        " ",
    ).title()

    factor_structure = str(
        settings[
            "factor_structure"
        ]
    ).replace(
        "_",
        " ",
    ).title()

    subtitle = (
        f"Base currency: {base_currency} | "
        f"Simulations: {int(settings['number_simulations']):,} | "
        f"Confidence: {confidence:.1%} | "
        f"Dependence: {dependence_model} | "
        f"Factor structure: {factor_structure}"
    )

    if (
        settings[
            "dependence_model"
        ]
        == "t_copula"
    ):

        subtitle += (
            f" | t degrees of freedom: "
            f"{settings['t_degrees_of_freedom']}"
        )

    story.append(
        Paragraph(
            subtitle,
            subtitle_style,
        )
    )

    # Main portfolio risk metrics.
    metric_data = [
        [
            "Total Exposure",
            "Default EL",
            f"{confidence:.1%} VaR",
            f"{confidence:.1%} ES",
            "Unexpected Loss",
            "Largest Default Loss",
        ],
        [
            f"{total_exposure:,.2f}",
            f"{summary['default_expected_loss']:,.2f}",
            f"{summary['value_at_risk']:,.2f}",
            f"{summary['expected_shortfall']:,.2f}",
            f"{summary['unexpected_loss']:,.2f}",
            f"{largest_default_loss:,.2f}",
        ],
        [
            base_currency,
            base_currency,
            base_currency,
            base_currency,
            base_currency,
            largest_counterparty,
        ],
    ]

    metric_table = Table(
        metric_data,
        colWidths=[
            4.35 * cm
        ] * 6,
        rowHeights=[
            0.65 * cm,
            0.75 * cm,
            0.55 * cm,
        ],
    )

    metric_table.setStyle(
        TableStyle([
            (
                "BACKGROUND",
                (0, 0),
                (-1, 0),
                colors.HexColor(
                    "#1F4E78"
                ),
            ),
            (
                "TEXTCOLOR",
                (0, 0),
                (-1, 0),
                colors.white,
            ),
            (
                "FONTNAME",
                (0, 0),
                (-1, 0),
                "Helvetica-Bold",
            ),
            (
                "FONTNAME",
                (0, 1),
                (-1, 1),
                "Helvetica-Bold",
            ),
            (
                "FONTSIZE",
                (0, 0),
                (-1, 0),
                7.5,
            ),
            (
                "FONTSIZE",
                (0, 1),
                (-1, 1),
                11,
            ),
            (
                "FONTSIZE",
                (0, 2),
                (-1, 2),
                7,
            ),
            (
                "TEXTCOLOR",
                (0, 2),
                (-1, 2),
                colors.HexColor(
                    "#555555"
                ),
            ),
            (
                "ALIGN",
                (0, 0),
                (-1, -1),
                "CENTER",
            ),
            (
                "VALIGN",
                (0, 0),
                (-1, -1),
                "MIDDLE",
            ),
            (
                "BOX",
                (0, 0),
                (-1, -1),
                0.5,
                colors.HexColor(
                    "#BFBFBF"
                ),
            ),
            (
                "INNERGRID",
                (0, 0),
                (-1, -1),
                0.25,
                colors.HexColor(
                    "#DDDDDD"
                ),
            ),
        ])
    )

    story.append(
        metric_table
    )

    story.append(
        Spacer(
            1,
            0.25 * cm,
        )
    )

    story.append(
        Paragraph(
            "Portfolio Loss Distribution",
            section_style,
        )
    )

    # Full loss distribution and tail zoom.
    loss_image = Image(
        str(
            loss_distribution_plot
        ),
        width=12.7 * cm,
        height=7.6 * cm,
    )

    tail_image = Image(
        str(
            tail_distribution_plot
        ),
        width=12.7 * cm,
        height=7.6 * cm,
    )

    distribution_table = Table(
        [
            [
                loss_image,
                tail_image,
            ]
        ],
        colWidths=[
            13.1 * cm,
            13.1 * cm,
        ],
    )

    distribution_table.setStyle(
        TableStyle([
            (
                "VALIGN",
                (0, 0),
                (-1, -1),
                "TOP",
            ),
            (
                "LEFTPADDING",
                (0, 0),
                (-1, -1),
                2,
            ),
            (
                "RIGHTPADDING",
                (0, 0),
                (-1, -1),
                2,
            ),
        ])
    )

    story.append(
        distribution_table
    )

    story.append(
        Spacer(
            1,
            0.2 * cm,
        )
    )

    story.append(
        Paragraph(
            (
                f"The largest deterministic single-counterparty "
                f"default loss is {largest_counterparty} at "
                f"{largest_default_loss:,.2f} {base_currency}. "
                f"The tail view highlights the relationship "
                f"between portfolio VaR, Expected Shortfall and "
                f"large single-name loss severity."
            ),
            note_style,
        )
    )

    story.append(
        PageBreak()
    )

    # =========================================================
    # Page 2 - Risk Drivers and Counterparty Overview
    # =========================================================

    story.append(
        Paragraph(
            "Risk Drivers and Counterparty Overview",
            title_style,
        )
    )

    # PDF-specific plots show only the eight largest exposures.
    report_exposure_plot = (
        create_report_exposure_plot(
            portfolio,
            base_currency,
            top_n=8,
        )
    )

    report_risk_plot = (
        create_report_risk_contribution_plot(
            portfolio,
            confidence,
            base_currency,
            top_n=8,
        )
    )

    exposure_image = Image(
        report_exposure_plot,
        width=12.7 * cm,
        height=6.8 * cm,
    )

    risk_image = Image(
        report_risk_plot,
        width=12.7 * cm,
        height=6.8 * cm,
    )

    chart_table = Table(
        [
            [
                exposure_image,
                risk_image,
            ]
        ],
        colWidths=[
            13.1 * cm,
            13.1 * cm,
        ],
    )

    chart_table.setStyle(
        TableStyle([
            (
                "VALIGN",
                (0, 0),
                (-1, -1),
                "TOP",
            ),
            (
                "LEFTPADDING",
                (0, 0),
                (-1, -1),
                2,
            ),
            (
                "RIGHTPADDING",
                (0, 0),
                (-1, -1),
                2,
            ),
        ])
    )

    story.append(
        chart_table
    )

    story.append(
        Spacer(
            1,
            0.15 * cm,
        )
    )

    story.append(
        Paragraph(
            "Counterparty Risk Summary",
            section_style,
        )
    )

    # Counterparty-level risk summary.
    counterparty_data = [
        [
            "Counterparty",
            "Credit Input",
            "PD",
            f"Exposure ({base_currency})",
            f"EL ({base_currency})",
            "Incremental VaR",
            "Incremental ES",
            "Marginal VaR",
            "Marginal ES",
        ]
    ]

    for _, row in portfolio.iterrows():

        if (
            row[
                "pd_method"
            ]
            == "rating"
        ):

            credit_input = str(
                row[
                    "rating"
                ]
            )

        else:

            credit_input = "Merton"

        counterparty_data.append([
            Paragraph(
                str(
                    row[
                        "counterparty"
                    ]
                ),
                table_text_style,
            ),
            credit_input,
            f"{row['pd']:.3e}",
            f"{row['exposure_base']:,.2f}",
            f"{row['expected_loss']:,.3f}",
            f"{row['incremental_var']:,.2f}",
            f"{row['incremental_es']:,.2f}",
            f"{row['marginal_var']:,.4f}",
            f"{row['marginal_es']:,.4f}",
        ])

    counterparty_table = Table(
        counterparty_data,
        colWidths=[
            4.3 * cm,
            1.9 * cm,
            2.0 * cm,
            2.7 * cm,
            2.2 * cm,
            2.7 * cm,
            2.7 * cm,
            2.6 * cm,
            2.6 * cm,
        ],
        repeatRows=1,
    )

    counterparty_table.setStyle(
        TableStyle([
            (
                "BACKGROUND",
                (0, 0),
                (-1, 0),
                colors.HexColor(
                    "#1F4E78"
                ),
            ),
            (
                "TEXTCOLOR",
                (0, 0),
                (-1, 0),
                colors.white,
            ),
            (
                "FONTNAME",
                (0, 0),
                (-1, 0),
                "Helvetica-Bold",
            ),
            (
                "FONTSIZE",
                (0, 0),
                (-1, 0),
                6.5,
            ),
            (
                "FONTSIZE",
                (0, 1),
                (-1, -1),
                7,
            ),
            (
                "ALIGN",
                (1, 1),
                (-1, -1),
                "RIGHT",
            ),
            (
                "ALIGN",
                (1, 0),
                (-1, 0),
                "CENTER",
            ),
            (
                "VALIGN",
                (0, 0),
                (-1, -1),
                "MIDDLE",
            ),
            (
                "GRID",
                (0, 0),
                (-1, -1),
                0.25,
                colors.HexColor(
                    "#CCCCCC"
                ),
            ),
            (
                "ROWBACKGROUNDS",
                (0, 1),
                (-1, -1),
                [
                    colors.white,
                    colors.HexColor(
                        "#F4F7FA"
                    ),
                ],
            ),
            (
                "TOPPADDING",
                (0, 0),
                (-1, -1),
                3,
            ),
            (
                "BOTTOMPADDING",
                (0, 0),
                (-1, -1),
                3,
            ),
        ])
    )

    story.append(
        counterparty_table
    )

    story.append(
        Spacer(
            1,
            0.2 * cm,
        )
    )

    story.append(
        Paragraph(
            "Key Model Considerations",
            section_style,
        )
    )

    considerations = [
        (
            "Portfolio exposures, ratings, LGDs and bond terms are "
            "demonstration inputs and do not represent actual company positions."
        ),
        (
            "Merton probabilities of default are structural model outputs. "
            "Very small model-implied PDs can occur and should be challenged "
            "for economic plausibility rather than interpreted as external ratings."
        ),
        (
            "FX rates are used to normalize exposures into the selected base "
            "currency; FX risk itself is not jointly simulated with credit risk."
        ),
        (
            "LGD is deterministic in the demonstration portfolio and the rating "
            "migration matrix is synthetic."
        ),
        (
            "Tail risk depends materially on the selected dependence model, "
            "factor structure and copula assumptions."
        ),
    ]

    for consideration in considerations:

        story.append(
            Paragraph(
                f"- {consideration}",
                note_style,
            )
        )

        story.append(
            Spacer(
                1,
                0.05 * cm,
            )
        )

    document.build(
        story
    )