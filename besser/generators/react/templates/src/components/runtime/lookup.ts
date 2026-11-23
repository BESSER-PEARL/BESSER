// A relationship to another class as a form edits it: a select of the related
// class's records. Shared by the table's create/edit dialog and bound forms.
import { apiRequest } from "../../api/generated/client";
import { extractRows } from "./bindingData";

export interface LookupTarget {
  // The related class
  entity?: string;
  // Its attribute shown for each record
  lookupField?: string;
  // Its identifying attribute (not necessarily `id`) and that attribute's type
  targetField?: string;
  targetType?: string;
}

/** The records of a class, as the options of a relationship field. */
export const fetchLookupOptions = async (entity: string): Promise<any[]> => {
  return extractRows(await apiRequest<unknown>("GET", `/${entity.toLowerCase()}/`));
};

// Canonical value of a related record: the target class's identifying
// attribute. Falls back to `id`, then to the display field, for older metadata.
export const lookupOptionValue = (target: LookupTarget, option: any): string => {
  const candidates = [target.targetField, "id", target.lookupField];
  for (const key of candidates) {
    if (key && option?.[key] !== undefined && option?.[key] !== null) {
      return String(option[key]);
    }
  }
  return "";
};

/** The text shown for a related record. */
export const lookupOptionLabel = (target: LookupTarget, option: any): string =>
  String((target.lookupField && option[target.lookupField]) || option.pages || option.stock || option.title || `ID: ${option.id}`);

// HTML selects store strings. Convert numeric keys only; e.g. Room.number='001'
// is a string identifier and must never become the different identifier 1.
export const coerceLookupTarget = (target: LookupTarget, value: any): string | number | null => {
  if (value === undefined || value === null || value === '') return null;
  const type = (target.targetType ?? 'int').toLowerCase();
  if (!['int', 'integer', 'float', 'double', 'number'].includes(type)) return String(value);
  const numeric = Number(value);
  if (!Number.isFinite(numeric) || (['int', 'integer'].includes(type) && !Number.isInteger(numeric))) return null;
  return numeric;
};
