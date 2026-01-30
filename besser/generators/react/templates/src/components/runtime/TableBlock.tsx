import React, { CSSProperties, useEffect, useState } from "react";
import axios from "axios";
import { TableComponent } from "../table/TableComponent";

export interface TableBlockProps {
  id: string;
  title?: string;
  options?: Record<string, any>;
  dataBinding?: Record<string, any>;
  styles?: CSSProperties;
}

const isNestedField = (field?: string): boolean => !!field && field.includes(".");

export const TableBlock: React.FC<TableBlockProps> = ({
  id,
  title,
  options,
  dataBinding,
  styles,
}) => {
  const [tableData, setTableData] = useState<any[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const endpoint = dataBinding?.endpoint;
    if (!endpoint) return;

    setLoading(true);
    setError(null);
    const backendBase = import.meta.env.VITE_API_URL || "http://localhost:8000";

    const hasLookupColumns = (options?.columns || []).some(
      (col: any) => typeof col === "object" && col.column_type === "lookup"
    );
    const hasNestedFields = isNestedField(dataBinding?.label_field) || isNestedField(dataBinding?.data_field);
    const detailed = hasLookupColumns || hasNestedFields;

    const urlParams = detailed ? "?detailed=true" : "";
    const url = endpoint.startsWith("/") ? backendBase + endpoint + urlParams : endpoint + urlParams;

    axios
      .get(url)
      .then((res) => {
        let data: any[] = [];
        if (Array.isArray(res.data)) {
          data = res.data;
        } else if (res.data && typeof res.data === "object") {
          const commonArrayKeys = ["data", "results", "items", "records", "list"];
          let foundKey = commonArrayKeys.find((key) => Array.isArray((res.data as any)[key]));
          if (!foundKey) {
            foundKey = Object.keys(res.data).find((key) => Array.isArray((res.data as any)[key]));
          }
          if (foundKey) {
            data = (res.data as any)[foundKey];
          }
        }
        setTableData(data);
      })
      .catch((err) => {
        console.error("[TableBlock] Error loading data:", err);
        setError("Error loading data");
        setTableData([]);
      })
      .finally(() => setLoading(false));
  }, [dataBinding?.endpoint, dataBinding?.label_field, dataBinding?.data_field, options?.columns]);

  if (loading) return <div id={id}>Loading data...</div>;
  if (error) return <div id={id}>{error}</div>;

  return (
    <TableComponent
      id={id}
      title={title}
      data={tableData}
      options={options || {}}
      styles={styles}
      dataBinding={dataBinding}
    />
  );
};
