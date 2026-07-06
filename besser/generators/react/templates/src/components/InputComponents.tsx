import React, { useState, useRef } from "react";

// ─── Alert ────────────────────────────────────────────────────────────────────

const _ALERT_COLORS: Record<string, { bg: string; border: string; text: string; icon: string }> = {
  info:    { bg: "#eff6ff", border: "#bfdbfe", text: "#1e40af", icon: "ℹ" },
  success: { bg: "#f0fdf4", border: "#bbf7d0", text: "#166534", icon: "✓" },
  warning: { bg: "#fffbeb", border: "#fde68a", text: "#92400e", icon: "⚠" },
  error:   { bg: "#fef2f2", border: "#fecaca", text: "#991b1b", icon: "✕" },
};

export interface AlertBlockProps {
  id?: string;
  className?: string;
  style?: React.CSSProperties;
  severity?: "info" | "success" | "warning" | "error";
  title?: string;
  content?: string;
  dismissible?: boolean;
}

export function AlertBlock({
  id, className, style, severity = "info", title, content, dismissible,
}: AlertBlockProps) {
  const [dismissed, setDismissed] = useState(false);
  if (dismissed) return null;
  const colors = _ALERT_COLORS[severity] ?? _ALERT_COLORS.info;
  return (
    <div
      id={id}
      className={className}
      role="alert"
      style={{
        display: "flex", alignItems: "flex-start", gap: "10px",
        padding: "10px 14px", borderRadius: "6px",
        border: `1px solid ${colors.border}`,
        background: colors.bg, color: colors.text, fontSize: "0.88em",
        marginBottom: "12px",
        ...style,
      }}
    >
      <span style={{ fontWeight: 700, flexShrink: 0 }}>{colors.icon}</span>
      <div style={{ flex: 1 }}>
        {title && <div style={{ fontWeight: 600, marginBottom: 2 }}>{title}</div>}
        <div>{content}</div>
      </div>
      {dismissible && (
        <button
          onClick={() => setDismissed(true)}
          style={{ background: "none", border: "none", cursor: "pointer", color: colors.text, padding: 0, lineHeight: 1 }}
          aria-label="Dismiss"
        >×</button>
      )}
    </div>
  );
}

// ─── Toggle ───────────────────────────────────────────────────────────────────

export interface ToggleInputProps {
  id?: string;
  name?: string;
  label?: string;
  defaultChecked?: boolean;
  disabled?: boolean;
  required?: boolean;
}

export function ToggleInput({
  id, name, label, defaultChecked = false, disabled, required,
}: ToggleInputProps) {
  const [checked, setChecked] = useState(defaultChecked);
  return (
    <div style={{ marginBottom: "4px" }}>
      <label style={{ display: "inline-flex", alignItems: "center", gap: "8px", cursor: disabled ? "default" : "pointer", userSelect: "none" }}>
        {label && (
          <span style={{ fontSize: "0.875em", fontWeight: 500, color: "#374151" }}>{label}</span>
        )}
        <span style={{ position: "relative", display: "inline-block", width: "44px", height: "24px", flexShrink: 0 }}>
          <input
            type="checkbox"
            id={id}
            name={name}
            checked={checked}
            disabled={disabled}
            required={required}
            style={{ opacity: 0, width: 0, height: 0, position: "absolute" }}
            onChange={(e) => setChecked(e.target.checked)}
          />
          <span style={{ position: "absolute", inset: 0, background: checked ? "#2563eb" : "#cbd5e1", borderRadius: "24px", transition: "background 0.2s", pointerEvents: "none" }} />
          <span style={{ position: "absolute", top: "3px", left: checked ? "23px" : "3px", width: "18px", height: "18px", borderRadius: "50%", background: "#fff", transition: "left 0.2s", boxShadow: "0 1px 3px rgba(0,0,0,0.2)", pointerEvents: "none" }} />
        </span>
      </label>
    </div>
  );
}

// ─── Slider ───────────────────────────────────────────────────────────────────

export interface SliderInputProps {
  id?: string;
  name?: string;
  min?: number;
  max?: number;
  step?: number;
  defaultValue?: number;
  disabled?: boolean;
}

export function SliderInput({
  id, name, min = 0, max = 100, step = 1, defaultValue, disabled,
}: SliderInputProps) {
  const [value, setValue] = useState<number>(defaultValue ?? min);
  return (
    <div style={{ display: "flex", alignItems: "center", gap: "10px" }}>
      <input
        type="range"
        id={id}
        name={name}
        min={min}
        max={max}
        step={step}
        value={value}
        disabled={disabled}
        style={{ flex: 1 }}
        onChange={(e) => setValue(Number(e.target.value))}
      />
      <span style={{ minWidth: "2.5em", textAlign: "right", fontSize: "0.875em", color: "#374151" }}>
        {value}
      </span>
    </div>
  );
}

// ─── Rating ───────────────────────────────────────────────────────────────────

export interface RatingInputProps {
  id?: string;
  name?: string;
  maxStars?: number;
  defaultValue?: number;
  disabled?: boolean;
}

export function RatingInput({
  id, name, maxStars = 5, defaultValue = 0, disabled,
}: RatingInputProps) {
  const [rating, setRating] = useState(defaultValue);
  const [hover, setHover] = useState(0);
  return (
    <div id={id} style={{ display: "flex", gap: "2px" }}>
      {Array.from({ length: maxStars }, (_, i) => i + 1).map((star) => (
        <button
          key={star}
          type="button"
          name={name}
          aria-label={`${star} star${star !== 1 ? "s" : ""}`}
          style={{
            background: "none", border: "none", padding: "0 2px",
            cursor: disabled ? "default" : "pointer", fontSize: "1.5em",
            color: star <= (hover || rating) ? "#f59e0b" : "#d1d5db",
            transition: "color 0.1s",
          }}
          disabled={disabled}
          onClick={() => setRating(star === rating ? 0 : star)}
          onMouseEnter={() => !disabled && setHover(star)}
          onMouseLeave={() => setHover(0)}
        >★</button>
      ))}
    </div>
  );
}

// ─── Tags ─────────────────────────────────────────────────────────────────────

export interface TagsInputProps {
  id?: string;
  name?: string;
  placeholder?: string;
  disabled?: boolean;
  required?: boolean;
}

export function TagsInput({
  id, name, placeholder = "Add tags…", disabled, required,
}: TagsInputProps) {
  const [tags, setTags] = useState<string[]>([]);
  const [inputValue, setInputValue] = useState("");

  const addTag = (raw: string) => {
    const tag = raw.trim();
    if (tag && !tags.includes(tag)) setTags((prev) => [...prev, tag]);
    setInputValue("");
  };

  const removeTag = (tag: string) => setTags((prev) => prev.filter((t) => t !== tag));

  return (
    <div
      id={id}
      style={{
        display: "flex", flexWrap: "wrap", alignItems: "center", gap: "4px",
        padding: "4px 8px", border: "1px solid #cbd5e1", borderRadius: "4px", minHeight: "38px",
      }}
    >
      {tags.map((tag) => (
        <span
          key={tag}
          style={{
            display: "inline-flex", alignItems: "center", gap: "4px",
            padding: "2px 8px", background: "#eff6ff", color: "#1d4ed8",
            borderRadius: "12px", fontSize: "0.82em",
          }}
        >
          {tag}
          <button
            type="button"
            onClick={() => removeTag(tag)}
            style={{ background: "none", border: "none", cursor: "pointer", color: "inherit", padding: 0, lineHeight: 1 }}
          >×</button>
        </span>
      ))}
      <input
        type="text"
        name={name}
        value={inputValue}
        placeholder={tags.length === 0 ? placeholder : undefined}
        disabled={disabled}
        required={required && tags.length === 0}
        style={{
          border: "none", outline: "none", fontFamily: "inherit",
          fontSize: "inherit", flex: "1 1 80px", minWidth: "80px", padding: "2px 0",
        }}
        onChange={(e) => setInputValue(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === "Enter" || e.key === ",") { e.preventDefault(); addTag(inputValue); }
          if (e.key === "Backspace" && !inputValue && tags.length) removeTag(tags[tags.length - 1]);
        }}
        onBlur={() => { if (inputValue) addTag(inputValue); }}
      />
    </div>
  );
}

// ─── OTP ──────────────────────────────────────────────────────────────────────

export interface OTPInputProps {
  id?: string;
  name?: string;
  length?: number;
  disabled?: boolean;
}

export function OTPInput({ id, name, length = 6, disabled }: OTPInputProps) {
  const safeLength = Math.max(2, Math.min(8, length));
  const [values, setValues] = useState<string[]>(() => Array(safeLength).fill(""));
  const refs = useRef<Array<HTMLInputElement | null>>(Array(safeLength).fill(null));

  const handleChange = (index: number, raw: string) => {
    const digit = raw.replace(/\D/g, "").slice(-1);
    const next = [...values];
    next[index] = digit;
    setValues(next);
    if (digit && index < safeLength - 1) refs.current[index + 1]?.focus();
  };

  const handleKeyDown = (index: number, e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === "Backspace" && !values[index] && index > 0) refs.current[index - 1]?.focus();
  };

  return (
    <div id={id} style={{ display: "flex", gap: "8px" }}>
      {values.map((v, i) => (
        <input
          key={i}
          ref={(r) => { refs.current[i] = r; }}
          type="text"
          name={i === 0 ? name : undefined}
          inputMode="numeric"
          maxLength={1}
          value={v}
          disabled={disabled}
          onChange={(e) => handleChange(i, e.target.value)}
          onKeyDown={(e) => handleKeyDown(i, e)}
          style={{
            width: "40px", height: "44px", textAlign: "center",
            border: "1px solid #cbd5e1", borderRadius: "6px",
            fontSize: "1.25em", fontFamily: "inherit",
          }}
        />
      ))}
    </div>
  );
}

// ─── DateRange ────────────────────────────────────────────────────────────────

export interface DateRangeInputProps {
  id?: string;
  name?: string;
  disabled?: boolean;
  required?: boolean;
}

export function DateRangeInput({ id, name, disabled, required }: DateRangeInputProps) {
  const [from, setFrom] = useState("");
  const [to, setTo] = useState("");
  const dateStyle: React.CSSProperties = {
    padding: "6px 10px", border: "1px solid #cbd5e1", borderRadius: "4px", fontFamily: "inherit",
  };
  return (
    <div id={id} style={{ display: "flex", alignItems: "center", gap: "8px", flexWrap: "wrap" }}>
      <input
        type="date"
        name={name ? `${name}_from` : undefined}
        value={from}
        max={to || undefined}
        disabled={disabled}
        required={required}
        style={dateStyle}
        onChange={(e) => setFrom(e.target.value)}
      />
      <span style={{ color: "#94a3b8", fontSize: "0.9em" }}>to</span>
      <input
        type="date"
        name={name ? `${name}_to` : undefined}
        value={to}
        min={from || undefined}
        disabled={disabled}
        required={required}
        style={dateStyle}
        onChange={(e) => setTo(e.target.value)}
      />
    </div>
  );
}

// ─── MultiSelect ──────────────────────────────────────────────────────────────

export interface MultiSelectOption {
  label: string;
  value: string;
}

export interface MultiSelectInputProps {
  id?: string;
  name?: string;
  options?: MultiSelectOption[];
  defaultValue?: string[];
  disabled?: boolean;
  required?: boolean;
}

export function MultiSelectInput({
  id, name, options = [], defaultValue = [], disabled,
}: MultiSelectInputProps) {
  const [selected, setSelected] = useState<string[]>(defaultValue);

  const toggle = (value: string) => {
    if (disabled) return;
    setSelected((prev) =>
      prev.includes(value) ? prev.filter((v) => v !== value) : [...prev, value]
    );
  };

  return (
    <div id={id} style={{ display: "flex", flexDirection: "column", gap: "6px" }}>
      {selected.length > 0 && (
        <div style={{ display: "flex", flexWrap: "wrap", gap: "4px" }}>
          {selected.map((val) => {
            const opt = options.find((o) => o.value === val);
            return (
              <span
                key={val}
                style={{
                  display: "inline-flex", alignItems: "center", gap: "4px",
                  padding: "2px 8px", background: "#eff6ff", color: "#1d4ed8",
                  borderRadius: "12px", fontSize: "0.82em",
                }}
              >
                {opt?.label ?? val}
                <button
                  type="button"
                  onClick={() => toggle(val)}
                  disabled={disabled}
                  style={{ background: "none", border: "none", cursor: "pointer", color: "inherit", padding: 0, lineHeight: 1 }}
                >×</button>
              </span>
            );
          })}
        </div>
      )}
      <div style={{ border: "1px solid #cbd5e1", borderRadius: "4px", overflow: "hidden" }}>
        {options.map((o, i) => {
          const isSelected = selected.includes(o.value);
          return (
            <div
              key={i}
              onClick={() => toggle(o.value)}
              style={{
                padding: "6px 10px",
                cursor: disabled ? "default" : "pointer",
                background: isSelected ? "#dbeafe" : i % 2 === 0 ? "#fff" : "#f8fafc",
                color: isSelected ? "#1d4ed8" : "#374151",
                fontWeight: isSelected ? 500 : 400,
                borderBottom: i < options.length - 1 ? "1px solid #e2e8f0" : "none",
                userSelect: "none" as const,
                display: "flex", justifyContent: "space-between", alignItems: "center",
              }}
            >
              <span>{o.label}</span>
              {isSelected && <span style={{ color: "#2563eb", fontWeight: 700 }}>✓</span>}
            </div>
          );
        })}
      </div>
      {/* hidden inputs for form submission */}
      {selected.map((val) => (
        <input key={val} type="hidden" name={name} value={val} />
      ))}
    </div>
  );
}
