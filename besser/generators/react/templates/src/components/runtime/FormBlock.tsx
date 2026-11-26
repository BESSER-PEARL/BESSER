import React, { CSSProperties, useState } from "react";
import { ApiError, apiRequest } from "../../api/generated/client";
import { notifyDataChanged } from "./dataVersion";

export interface FormBlockField {
  // Name of the control in the form, the attribute it sets, and its type
  name: string;
  field: string;
  type?: string;
}

export interface FormBlockProps {
  id?: string;
  endpoint: string;
  entity?: string;
  fields: FormBlockField[];
  submitLabel?: string;
  // Design classes of the form's own submit button
  submitClassName?: string;
  className?: string;
  style?: CSSProperties;
  children?: React.ReactNode;
}

const toValue = (type: string | undefined, raw: FormDataEntryValue | null): any => {
  const text = typeof raw === "string" ? raw : "";
  switch ((type || "str").toLowerCase()) {
    case "int":
    case "integer":
      return parseInt(text, 10);
    case "float":
    case "double":
    case "decimal":
      return parseFloat(text);
    case "bool":
    case "boolean":
      return raw !== null && text !== "false";
    default:
      return text;
  }
};

const errorText = (err: unknown): string => {
  if (err instanceof ApiError && err.status > 0) {
    const data = err.body as any;
    const detail = data?.detail;
    if (Array.isArray(detail)) {
      return detail.map((e: any) => `${e.loc ? e.loc[e.loc.length - 1] : "field"}: ${e.msg}`).join("; ");
    }
    if (typeof detail === "string") return detail;
    if (detail) return JSON.stringify(detail);
    return data?.message || "Could not save. Please check your input.";
  }
  return "Network error. Please try again.";
};

/**
 * A GUI form bound to a class: submitting it creates a record (POST to the
 * class endpoint) from the controls bound to the class's attributes.
 */
export const FormBlock: React.FC<FormBlockProps> = ({
  id,
  endpoint,
  entity,
  fields,
  submitLabel = "Submit",
  submitClassName,
  className,
  style,
  children,
}) => {
  const [saving, setSaving] = useState(false);
  const [status, setStatus] = useState<{ ok: boolean; text: string } | null>(null);

  const handleSubmit = async (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    const form = event.currentTarget;
    const data = new FormData(form);
    const payload: Record<string, any> = {};
    for (const field of fields) {
      const raw = data.get(field.name);
      const isBool = ["bool", "boolean"].includes((field.type || "").toLowerCase());
      // Left empty: omitted, so the backend applies the default or reports it missing
      if (!isBool && (raw === null || raw === "")) continue;
      const value = toValue(field.type, raw);
      if (typeof value === "number" && Number.isNaN(value)) continue;
      payload[field.field] = value;
    }
    setSaving(true);
    setStatus(null);
    try {
      await apiRequest("POST", endpoint, { body: payload });
      notifyDataChanged();
      form.reset();
      setStatus({ ok: true, text: `${entity || "Record"} saved.` });
    } catch (err) {
      console.error("[FormBlock] Error saving:", err);
      setStatus({ ok: false, text: errorText(err) });
    } finally {
      setSaving(false);
    }
  };

  return (
    <form id={id} className={className} style={style} onSubmit={handleSubmit}>
      {children}
      <button type="submit" className={submitClassName} disabled={saving}>
        {saving ? "Saving..." : submitLabel}
      </button>
      {status && (
        <div
          role={status.ok ? "status" : "alert"}
          style={{
            marginTop: "10px",
            padding: "8px 12px",
            borderRadius: "6px",
            fontSize: "14px",
            background: status.ok ? "#dcfce7" : "#fee2e2",
            color: status.ok ? "#166534" : "#991b1b",
          }}
        >
          {status.text}
        </div>
      )}
    </form>
  );
};
