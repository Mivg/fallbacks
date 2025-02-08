def format_float_value(x, is_bold, format_spec=".3f"):
    """Helper function to format float values and apply bold if needed."""
    if pd.isna(x) or x in {"MISSING", "-"}:
        return x
    return f"\\textbf{{{x:{format_spec}}}}" if is_bold else f"{x:{format_spec}}"


def save_to_latex(
    df, filename, label, caption, group_by=None, bold=False, float_cols=None, lower_is_better=None, lines_after=None, format_spec=".3f", tabular_only=False,
):
    if group_by and group_by in df.columns:
        df[group_by] = df[group_by].astype(str)  # Ensure the group_by column is a string for text manipulation
        df.set_index(group_by, inplace=True)
        df = df.reset_index(drop=False)
        counts = df.groupby(group_by).size().to_dict()

    if float_cols:
        if group_by:
            for group_val in df[group_by].unique():
                subdf = df[df[group_by] == group_val]
                format_cols_in_subdf(subdf, float_cols, lower_is_better, format_spec, bold=bold)
                df.loc[df[group_by] == group_val, float_cols] = subdf[float_cols]
        else:
            format_cols_in_subdf(df, float_cols, lower_is_better, format_spec, bold=bold)

    num_columns = len(df.reset_index().columns) - 1
    column_format = "l" + "c" * (num_columns - 1)

    params = {
        "index": False,
        "column_format": column_format,
        "escape": False,
    }

    latex_content = df.to_latex(**params)

    # Modify LaTeX content for custom table setup
    lines = latex_content.splitlines()
    body_start = next(i for i, line in enumerate(lines) if "\\toprule" in line or "\\midrule" in line) + 1
    body_end = next(i for i, line in enumerate(lines) if "\\bottomrule" in line)
    lines = [l for l in lines[body_start:body_end] if l != "\\midrule"]
    if group_by:
        curr_val = None
        for i, line in enumerate(lines[1:]):
            group_val, row = line.split("&", 1)
            group_val = group_val.strip()
            row = "& " + row
            if curr_val != group_val:
                curr_val = group_val
                lines[i + 1] = "\\midrule\n" + f"\\multirow{{{counts.pop(group_val)}}}{{*}}{{{group_val}}}" + row
            else:
                lines[i + 1] = row
    else:
        lines.insert(1, "\\midrule")

    if lines_after:
        for i in lines_after:
            lines.insert(i + 2, "\\midrule")
    table_body = "\n".join(lines)  # Extract only the body content

    output = ""
    if not tabular_only:
        # Write the custom table setup including caption, label, and row colors
        output += "\\begin{table}[ht]\n"
        # f.write('\\rowcolors{2}{gray!25}{white}\n')
        output += "\\centering\n"
        output += "\\caption{" + caption + "}\n"
        output += "\\label{" + label + "}\n"
        output += "\\resizebox{\linewidth}{!}{% Resize table to fit the linewidth\n"
    output += "\\begin{tabular}{" + column_format + "}\n"
    output += "\\toprule\n"
    output += table_body
    output += "\\bottomrule\n"
    output += "\\end{tabular}\n"
    if not tabular_only:
        output += "}\n"  # End resizebox
        output += "\\end{table}\n"
    if filename:
        with open(filename, "w") as f:
            f.write(output)
    return output


def format_cols_in_subdf(df, float_cols, lower_is_better, format_spec=".3f", bold=False):
    for col in float_cols:
        try:
            valid_values = df[col].replace({"MISSING", "-"}, pd.NA).dropna().astype(float)
            # TODO - account for rounding?
            best_value = valid_values.min() if (lower_is_better and col in lower_is_better) else valid_values.max()
            df[col] = df[col].apply(lambda x: format_float_value(x, x == best_value and bold, format_spec))
        except Exception as e:
            print(f"Error processing column {col}: {e}")