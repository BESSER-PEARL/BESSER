export interface StyleData {
  selectors: string[];
  style: Record<string, string | number>;
}

// Optionally pass styleId to match by style_id as well
export function applyStyle(selector: string, styles: StyleData[], styleId?: string) {
  let styleObj = styles.find((s) => s.selectors.includes(selector));
  if (!styleObj && styleId) {
    styleObj = styles.find((s) => s.selectors.includes(`#${styleId}`));
  }
  return styleObj ? styleObj.style : {};
}