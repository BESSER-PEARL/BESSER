import React, { CSSProperties, useEffect, useState } from "react";
import { useDataVersion } from "./dataVersion";
import { LookupTarget, fetchLookupOptions, lookupOptionLabel, lookupOptionValue } from "./lookup";

export interface LookupSelectProps extends LookupTarget {
  id?: string;
  // Name of the control in its form
  name: string;
  entity: string;
  required?: boolean;
  className?: string;
  style?: CSSProperties;
}

/**
 * A form's select of the records of a related class (e.g. the customer of a
 * new order). Its value is the chosen record's identifying attribute.
 */
export const LookupSelect: React.FC<LookupSelectProps> = ({ id, name, required, className, style, ...target }) => {
  const [options, setOptions] = useState<any[]>([]);
  const dataVersion = useDataVersion();

  useEffect(() => {
    let current = true;
    fetchLookupOptions(target.entity)
      .then((rows) => {
        if (current) setOptions(rows);
      })
      .catch((err) => {
        console.error(`[LookupSelect] Error loading ${target.entity} options:`, err);
        if (current) setOptions([]);
      });
    return () => {
      current = false;
    };
  }, [target.entity, dataVersion]);

  return (
    <select id={id} name={name} required={required} defaultValue="" className={className} style={style}>
      <option value="">-- Select {target.entity} --</option>
      {options.map((option) => (
        <option key={lookupOptionValue(target, option)} value={lookupOptionValue(target, option)}>
          {lookupOptionLabel(target, option)}
        </option>
      ))}
    </select>
  );
};
