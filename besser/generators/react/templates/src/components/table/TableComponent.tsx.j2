import React, { CSSProperties, useEffect, useMemo, useState } from "react";
import axios from "axios";
import { createPortal } from "react-dom";
import { useTableContext } from "../../contexts/TableContext";
import { ColumnFilter } from "./ColumnFilter";
import { ColumnSort } from "./ColumnSort";
import "./TableComponent.css";

interface TableColumn {
  field: string;
  label: string;
  type?: string;
  columnType?: string;
  path?: string;
  entity?: string;
  options?: string[];
  required?: boolean;
  lookupField?: string;
  relationshipKey?: string;
}

interface TableOptions {
  showHeader?: boolean;
  stripedRows?: boolean;
  showPagination?: boolean;
  rowsPerPage?: number;
  columns?: Array<{ field: string; label?: string; type?: string; column_type?: string; path?: string } | string>;
}

interface Props {
  id: string;
  title?: string;
  data?: any[];
  options?: TableOptions;
  styles?: CSSProperties;
  dataBinding?: Record<string, any>;
}

const humanize = (value: string): string => {
  if (!value) {
    return "";
  }
  return value
    .replace(/[_-]+/g, " ")
    .replace(/([a-z])([A-Z])/g, "$1 $2")
    .replace(/\s+/g, " ")
    .trim()
    .replace(/^./, (str) => str.toUpperCase());
};

const formatCellValue = (value: any): string => {
  if (value === null || value === undefined) {
    return "";
  }
  if (typeof value === "string") {
    return value;
  }
  if (typeof value === "number" || typeof value === "boolean") {
    return String(value);
  }
  if (value instanceof Date) {
    return value.toLocaleString();
  }
  return JSON.stringify(value);
};

// Helper to get value from nested path (e.g., customer.age)
const getNestedValue = (obj: any, path: string): any => {
  return path.split('.').reduce((current, key) => current?.[key], obj);
};

export const TableComponent: React.FC<Props> = ({
  id,
  title,
  data,
  options,
  styles,
  dataBinding,
}) => {
  // Local state for table data
  const [tableData, setTableData] = useState<any[]>(data ?? []);
  const { setSelectedRow, registerTableRefresh } = useTableContext();
  const [selectedRowIndex, setSelectedRowIndex] = useState<number | null>(null);
  const [columnFilters, setColumnFilters] = useState<Record<string, string>>({});
  const [sortConfig, setSortConfig] = useState<{ field: string; direction: 'asc' | 'desc' } | null>(null);

  // Fetch table data from backend
  const fetchTableData = async () => {
    const endpoint = dataBinding?.endpoint
      ? dataBinding.endpoint
      : dataBinding?.entity
        ? `/${dataBinding.entity}/`
        : id
          ? `/${id}/`
          : "";
    if (!endpoint) return;
    const backendBase = process.env.REACT_APP_API_URL || "http://localhost:8000";
    
    // Check if table has lookup columns - if so, request detailed data with joins
    const hasLookupColumns = options?.columns?.some(
      (col: any) => typeof col === 'object' && col.column_type === 'lookup'
    );
    
    // Add detailed=true query param if there are lookup columns
    const urlParams = hasLookupColumns ? '?detailed=true' : '';
    const url = endpoint.startsWith("/") 
      ? backendBase + endpoint + urlParams
      : endpoint + urlParams;
    
    try {
      const response = await axios.get(url);
      if (Array.isArray(response.data)) {
        setTableData(response.data);
      } else if (response.data && typeof response.data === "object") {
        setTableData(Array.isArray(response.data.results) ? response.data.results : [response.data]);
      }
    } catch (err) {
      console.error("Error fetching table data:", err);
    }
  };

  // Register refresh function with context
  useEffect(() => {
    registerTableRefresh(id, fetchTableData);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [id]);

  // Initial fetch and update on data prop change
  useEffect(() => {
    if (data && Array.isArray(data)) {
      setTableData(data);
    } else {
      fetchTableData();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [data]);
  const resolvedOptions: Required<TableOptions> & { actionButtons: boolean } = {
    showHeader: options?.showHeader ?? true,
    stripedRows: options?.stripedRows ?? false,
    showPagination: options?.showPagination ?? true,
    rowsPerPage: options?.rowsPerPage ?? 5,
    columns: options?.columns ?? [],
    actionButtons: (options as any)?.actionButtons ?? (options as any)?.["action-buttons"] ?? false,
  };

  const normalizedRows = useMemo(() => {
    if (!Array.isArray(tableData)) {
      return [];
    }
    return tableData
      .filter((row) => row !== null && row !== undefined)
      .map((row) => {
        if (row && typeof row === "object" && !Array.isArray(row)) {
          return row;
        }
        return { value: row };
      });
  }, [tableData]);

  const columns: TableColumn[] = useMemo(() => {
    if (resolvedOptions.columns.length > 0) {
      return resolvedOptions.columns
        .filter((col): col is { field: string; label?: string; type?: string; column_type?: string; path?: string; entity?: string; options?: string[]; required?: boolean } | string => Boolean(col))
        .map((col) => {
          if (typeof col === "string") {
            return { field: col, label: humanize(col), type: "string" };
          }
          // For lookup columns, use path as the actual field to read
          const actualField = col.column_type === "lookup" && col.path
            ? col.path  // Use path for lookup (e.g., "customer")
            : col.field;
          
          // The relationship key in the API response is simply the path name
          // Example: "customer"
          const relationshipKey = col.column_type === "lookup" && col.path
            ? col.path
            : undefined;
          
          return {
            field: actualField,
            label: humanize(col.label || col.field),
            type: col.type || "string",
            columnType: col.column_type,
            path: col.path,
            entity: col.entity,
            options: col.options,
            required: col.required ?? false,
            lookupField: col.column_type === "lookup" ? col.field : undefined,  // Store the nested field (e.g., "name")
            relationshipKey: relationshipKey,  // API relationship key (e.g., "customer")
          };
        });
    }

    if (normalizedRows.length === 0) {
      return [];
    }

    const columnOrder: string[] = [];
    normalizedRows.forEach((row) => {
      Object.keys(row || {}).forEach((key) => {
        if (!columnOrder.includes(key)) {
          columnOrder.push(key);
        }
      });
    });

    return columnOrder.map((field) => ({
      field,
      label: humanize(field),
      type: "string",
    }));
  }, [normalizedRows, resolvedOptions.columns]);

  // Apply column filters
  const filteredRows = useMemo(() => {
    if (Object.keys(columnFilters).length === 0) {
      return normalizedRows;
    }

    return normalizedRows.filter((row) => {
      return Object.entries(columnFilters).every(([field, filterValue]) => {
        if (!filterValue) return true;

        // Find the column definition to check if it's a lookup
        const column = columns.find(col => col.field === field);
        let cellValue;

        // Handle lookup columns
        if (column?.columnType === 'lookup' && column.lookupField) {
          if (column.type === 'list') {
            // For list lookups (e.g., books array), get array of nested values
            const relatedArray = row[column.field];
            if (Array.isArray(relatedArray)) {
              // Extract all lookup field values and join them for filtering
              cellValue = relatedArray
                .map(item => item[column.lookupField!])
                .filter(val => val !== null && val !== undefined)
                .join(', ');
            } else {
              cellValue = '';
            }
          } else {
            // For single lookups (e.g., library.name), get nested value
            if (column.relationshipKey) {
              cellValue = getNestedValue(row, `${column.relationshipKey}.${column.lookupField}`);
            }
            if (!cellValue) {
              cellValue = getNestedValue(row, `${column.field}.${column.lookupField}`);
            }
          }
        } else {
          cellValue = row[field];
        }

        // Special operators that don't need a value
        if (filterValue === "empty") {
          return cellValue === null || cellValue === undefined || cellValue === "";
        }
        if (filterValue === "not_empty") {
          return cellValue !== null && cellValue !== undefined && cellValue !== "";
        }

        // Parse operator and value (format: "operator:value")
        const parts = filterValue.split(":");
        const operator = parts[0];
        const searchValue = parts.slice(1).join(":");

        if (!searchValue && operator !== "empty" && operator !== "not_empty") {
          // Backward compatibility: no operator means contains
          return String(cellValue || "").toLowerCase().includes(filterValue.toLowerCase());
        }

        // Handle null/undefined
        if (cellValue === null || cellValue === undefined) {
          return false;
        }

        const cellStr = String(cellValue).toLowerCase();
        const searchStr = searchValue.toLowerCase();

        // String operators
        if (operator === "contains") {
          return cellStr.includes(searchStr);
        }
        if (operator === "equals") {
          return cellStr === searchStr;
        }
        if (operator === "starts_with") {
          return cellStr.startsWith(searchStr);
        }
        if (operator === "ends_with") {
          return cellStr.endsWith(searchStr);
        }

        // Numeric/Date operators
        const cellNum = Number(cellValue);
        const searchNum = Number(searchValue);

        // Try numeric comparison first
        if (!isNaN(cellNum) && !isNaN(searchNum)) {
          if (operator === "eq") return cellNum === searchNum;
          if (operator === "ne") return cellNum !== searchNum;
          if (operator === "gt") return cellNum > searchNum;
          if (operator === "gte") return cellNum >= searchNum;
          if (operator === "lt") return cellNum < searchNum;
          if (operator === "lte") return cellNum <= searchNum;
        }

        // Try date comparison
        const cellDate = new Date(cellValue);
        const searchDate = new Date(searchValue);
        
        if (!isNaN(cellDate.getTime()) && !isNaN(searchDate.getTime())) {
          if (operator === "eq") return cellDate.getTime() === searchDate.getTime();
          if (operator === "ne") return cellDate.getTime() !== searchDate.getTime();
          if (operator === "gt") return cellDate.getTime() > searchDate.getTime();
          if (operator === "gte") return cellDate.getTime() >= searchDate.getTime();
          if (operator === "lt") return cellDate.getTime() < searchDate.getTime();
          if (operator === "lte") return cellDate.getTime() <= searchDate.getTime();
        }

        // String comparison as fallback
        if (operator === "eq") return cellStr === searchStr;
        if (operator === "ne") return cellStr !== searchStr;
        if (operator === "gt") return cellStr > searchStr;
        if (operator === "gte") return cellStr >= searchStr;
        if (operator === "lt") return cellStr < searchStr;
        if (operator === "lte") return cellStr <= searchStr;

        return false;
      });
    });
  }, [normalizedRows, columnFilters]);

  const handleFilterChange = (field: string, value: string) => {
    setColumnFilters((prev) => {
      if (!value) {
        const newFilters = { ...prev };
        delete newFilters[field];
        return newFilters;
      }
      return { ...prev, [field]: value };
    });
    setCurrentPage(1); // Reset to first page when filtering
  };

  const handleSortChange = (field: string) => {
    setSortConfig((prev) => {
      if (!prev || prev.field !== field) {
        return { field, direction: 'asc' };
      }
      if (prev.direction === 'asc') {
        return { field, direction: 'desc' };
      }
      return null; // Remove sorting
    });
  };

  // Apply sorting after filtering
  const sortedAndFilteredRows = useMemo(() => {
    if (!sortConfig) {
      return filteredRows;
    }

    const sorted = [...filteredRows].sort((a, b) => {
      const aValue = a[sortConfig.field];
      const bValue = b[sortConfig.field];

      // Handle null/undefined
      if (aValue === null || aValue === undefined) return 1;
      if (bValue === null || bValue === undefined) return -1;

      // Detect type and sort accordingly
      const aNum = Number(aValue);
      const bNum = Number(bValue);

      // If both are valid numbers, sort numerically
      if (!isNaN(aNum) && !isNaN(bNum)) {
        return sortConfig.direction === 'asc' ? aNum - bNum : bNum - aNum;
      }

      // Try date parsing
      const aDate = new Date(aValue);
      const bDate = new Date(bValue);
      if (!isNaN(aDate.getTime()) && !isNaN(bDate.getTime())) {
        return sortConfig.direction === 'asc' 
          ? aDate.getTime() - bDate.getTime() 
          : bDate.getTime() - aDate.getTime();
      }

      // Default to string comparison
      const aStr = String(aValue).toLowerCase();
      const bStr = String(bValue).toLowerCase();
      
      if (sortConfig.direction === 'asc') {
        return aStr < bStr ? -1 : aStr > bStr ? 1 : 0;
      } else {
        return bStr < aStr ? -1 : bStr > aStr ? 1 : 0;
      }
    });

    return sorted;
  }, [filteredRows, sortConfig]);

  const pageSize = Math.max(
    1,
    Number.isFinite(Number(resolvedOptions.rowsPerPage))
      ? Number(resolvedOptions.rowsPerPage)
      : 5
  );

  const [currentPage, setCurrentPage] = useState(1);
  const [showModal, setShowModal] = useState(false);
  const [formValues, setFormValues] = useState<Record<string, any>>({});
  const [modalMode, setModalMode] = useState<'add' | 'edit'>('add');
  const [editRowData, setEditRowData] = useState<any>(null);
  const [lookupOptions, setLookupOptions] = useState<Record<string, any[]>>({});
  const [validationError, setValidationError] = useState<string>("");

  useEffect(() => {
    setCurrentPage(1);
  }, [pageSize, filteredRows.length]);

  // Reset form values when modal opens
  useEffect(() => {
    if (showModal) {
      setValidationError("");
      const initialValues: Record<string, any> = {};
      columns.forEach(col => {
        if (modalMode === 'edit' && editRowData) {
          // For lookup columns in edit mode, use the FK ID
          if (col.columnType === 'lookup') {
            if (col.type === 'list') {
              // For list-type lookups (1:N or N:M), get array of IDs from related objects
              const relatedArray = editRowData[col.relationshipKey || col.field];
              if (Array.isArray(relatedArray)) {
                initialValues[col.field] = relatedArray.map((item: any) => String(item.id));
              } else {
                initialValues[col.field] = [];
              }
            } else {
              // For single lookups (N:1), use the path as the field name
              const fkField = col.path || col.field;
              const fkIdField = `${fkField}_id`;
              initialValues[col.field] = editRowData[fkIdField] ?? "";
            }
          } else {
            const value = editRowData[col.field];
            if (col.type === 'list') {
              initialValues[col.field] = Array.isArray(value) ? value.join(', ') : (value ?? "");
            } else {
              initialValues[col.field] = value ?? "";
            }
          }
        } else {
          // For new records, initialize appropriately
          if (col.columnType === 'lookup' && col.type === 'list') {
            initialValues[col.field] = [];
          } else {
            initialValues[col.field] = "";
          }
        }
      });
      setFormValues(initialValues);

      // Fetch lookup options for lookup columns
      const fetchLookupOptions = async () => {
        const options: Record<string, any[]> = {};
        const backendBase = process.env.REACT_APP_API_URL || "http://localhost:8000";
        
        for (const col of columns) {
          if (col.columnType === 'lookup' && col.entity) {
            const endpoint = col.entity.toLowerCase();
            try {
              const response = await axios.get(`${backendBase}/${endpoint}/`);
              if (Array.isArray(response.data)) {
                options[endpoint] = response.data;
              } else if (response.data && typeof response.data === 'object') {
                options[endpoint] = Array.isArray(response.data.results) ? response.data.results : [response.data];
              }
            } catch (err) {
              console.error(`Error fetching ${endpoint} options:`, err);
              options[endpoint] = [];
            }
          }
        }
        setLookupOptions(options);
      };

      fetchLookupOptions();
    }
  }, [showModal, columns, modalMode, editRowData]);

  const totalPages = resolvedOptions.showPagination
    ? Math.max(1, Math.ceil(sortedAndFilteredRows.length / pageSize))
    : 1;

  const visibleRows = useMemo(() => {
    if (!resolvedOptions.showPagination) {
      return sortedAndFilteredRows;
    }
    const start = (currentPage - 1) * pageSize;
    return sortedAndFilteredRows.slice(start, start + pageSize);
  }, [currentPage, sortedAndFilteredRows, pageSize, resolvedOptions.showPagination]);

  const containerStyle: CSSProperties = {
    display: "flex",
    flexDirection: "column",
    gap: "16px",
    backgroundColor: "#ffffff",
    borderRadius: "8px",
    boxShadow: "0 2px 8px rgba(0,0,0,0.08)",
    padding: "20px",
    width: "100%",
    maxWidth: "100%",
    minWidth: 0,
    alignSelf: "stretch",
    boxSizing: "border-box",
    overflow: "hidden",
    ...styles,
  };

  const tableStyle: CSSProperties = {
    borderCollapse: "collapse",
    fontFamily: "Inter, system-ui, -apple-system, BlinkMacSystemFont, sans-serif",
    fontSize: "14px",
    width: "100%",
    tableLayout: "auto",
  };

  // Helper to get endpoint for entity
  const getEndpoint = () => {
    let endpoint = "";
    if (dataBinding?.endpoint) {
      endpoint = dataBinding.endpoint;
    } else if (dataBinding?.entity) {
      endpoint = `/${dataBinding.entity}/`;
    }
    if (!endpoint && id) {
      endpoint = `/${id}/`;
    }
    return endpoint;
  };

  // Helper to get row id
  const getRowId = (row: any) => {
    return row?.id ?? row?.ID ?? row?.Id ?? Object.values(row)[0];
  };

  return (
    <div id={id} style={containerStyle} className="table-wrapper">
      {title && (
        <h3 style={{ margin: 0, color: "#1e293b", fontSize: "18px" }}>{title}</h3>
      )}

      {/* Action Buttons above table */}
      {resolvedOptions.actionButtons && (
        <div style={{ display: "flex", justifyContent: "flex-end", marginBottom: "8px" }}>
          <button
            style={{
              padding: '6px 14px',
              background: 'linear-gradient(90deg, #2563eb 0%, #1e40af 100%)',
              color: '#fff',
              border: 'none',
              borderRadius: '4px',
              fontWeight: 600,
              cursor: 'pointer',
              fontSize: '13px',
              boxShadow: '0 1px 4px rgba(37,99,235,0.10)',
              letterSpacing: '0.01em',
              transition: 'background 0.2s',
              marginRight: '0',
              display: 'flex',
              alignItems: 'center',
              gap: '6px',
            }}
            type="button"
            title={`Add ${dataBinding?.entity || 'Register'}`}
            onClick={() => {
              setModalMode('add');
              setEditRowData(null);
              setShowModal(true);
            }}
          >
            <svg width="16" height="16" viewBox="0 0 20 20" fill="none" xmlns="http://www.w3.org/2000/svg">
              <circle cx="10" cy="10" r="8" fill="#2563eb"/>
              <rect x="9" y="5.5" width="2" height="9" rx="1" fill="white"/>
              <rect x="5.5" y="9" width="9" height="2" rx="1" fill="white"/>
            </svg>
            {`Add ${dataBinding?.entity || 'Register'}`}
          </button>
        </div>
      )}

      {/* Modal for Add/Edit Register */}
      {showModal && (
        createPortal(
          <div style={{
            position: "fixed",
            top: 0,
            left: 0,
            width: "100vw",
            height: "100vh",
            background: "rgba(30,41,59,0.25)",
            zIndex: 2147483647,
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
          }}>
            <div style={{
              background: "#fff",
              borderRadius: "12px",
              boxShadow: "0 4px 24px rgba(30,41,59,0.18)",
              padding: "32px 24px 24px 24px",
              minWidth: "320px",
              maxWidth: "90vw",
              width: "400px",
              position: "relative",
            }}>
              <h4 style={{ margin: 0, marginBottom: "18px", color: "#1e293b" }}>
                {modalMode === 'edit' ? `Edit ${dataBinding?.entity || 'Register'}` : `Add ${dataBinding?.entity || 'Register'}`}
              </h4>
              {validationError && (
                <div style={{
                  padding: "12px",
                  marginBottom: "16px",
                  backgroundColor: "#fee2e2",
                  border: "1px solid #fecaca",
                  borderRadius: "6px",
                  color: "#991b1b",
                  fontSize: "14px",
                }}>
                  {validationError}
                </div>
              )}
              <form
                onSubmit={async e => {
                  e.preventDefault();
                  
                  // Validate required fields
                  const missingFields: string[] = [];
                  columns.forEach(col => {
                    if (col.required) {
                      if (col.columnType === 'lookup') {
                        if (col.type === 'list') {
                          const value = formValues[col.field];
                          if (!value || (Array.isArray(value) && value.length === 0)) {
                            missingFields.push(col.label);
                          }
                        } else {
                          const value = formValues[col.field];
                          if (!value || value === "") {
                            missingFields.push(col.label);
                          }
                        }
                      } else {
                        const value = formValues[col.field];
                        if (value === undefined || value === null || value === "") {
                          missingFields.push(col.label);
                        }
                      }
                    }
                  });
                  
                  if (missingFields.length > 0) {
                    setValidationError(`Please fill in the following required fields: ${missingFields.join(', ')}`);
                    return;
                  }
                  
                  setValidationError("");
                  const endpoint = getEndpoint();
                  const backendBase = process.env.REACT_APP_API_URL || "http://localhost:8000";
                  
                  // Process form values based on column types
                  const processedValues: Record<string, any> = {};
                  columns.forEach(col => {
                    // For lookup columns, use the path as the API field name
                    if (col.columnType === 'lookup' && col.path) {
                      if (col.type === 'list') {
                        // For list-type lookups (N:M or 1:N), send array of IDs
                        const value = formValues[col.field];
                        processedValues[col.path] = Array.isArray(value) 
                          ? value.map(v => parseInt(v, 10)).filter(v => !isNaN(v))
                          : [];
                      } else {
                        // For single lookups (N:1), send single FK ID using the field from form
                        const value = formValues[col.field];
                        processedValues[col.path] = (value && value !== "") ? parseInt(value, 10) : null;
                      }
                    } else {
                      const value = formValues[col.field];
                      if (col.type === 'list') {
                        // Convert comma-separated string to array
                        processedValues[col.field] = typeof value === 'string' 
                          ? value.split(',').map(item => item.trim()).filter(item => item !== '')
                          : (Array.isArray(value) ? value : []);
                      } else if (col.type === 'int') {
                        // Convert to integer
                        const intValue = parseInt(value, 10);
                        processedValues[col.field] = isNaN(intValue) ? 0 : intValue;
                      } else if (col.type === 'float') {
                        // Convert to float
                        const floatValue = parseFloat(value);
                        processedValues[col.field] = isNaN(floatValue) ? 0 : floatValue;
                      } else if (col.type === 'bool' || col.type === 'boolean') {
                        // Convert to boolean
                        processedValues[col.field] = Boolean(value);
                      } else {
                        // Keep as string (includes date, datetime, time, text)
                        processedValues[col.field] = value;
                      }
                    }
                  });
                  
                  if (modalMode === 'add') {
                    const url = endpoint.startsWith("/") ? backendBase + endpoint : endpoint;
                    try {
                      await axios.post(url, processedValues);
                      await fetchTableData();
                      setShowModal(false);
                    } catch (err) {
                      console.error("Error saving data:", err);
                      if (axios.isAxiosError(err) && err.response) {
                        console.error("Response data:", err.response.data);
                        // Parse and display validation errors from backend (Pydantic/FastAPI)
                        const detail = err.response.data?.detail;
                        if (detail) {
                          if (Array.isArray(detail)) {
                            const errorMessages = detail.map((e: any) => {
                              const field = e.loc ? e.loc[e.loc.length - 1] : 'field';
                              return `${field}: ${e.msg}`;
                            }).join('; ');
                            setValidationError(errorMessages);
                          } else if (typeof detail === 'string') {
                            setValidationError(detail);
                          } else {
                            setValidationError(JSON.stringify(detail));
                          }
                        } else if (err.response.data?.message) {
                          setValidationError(err.response.data.message);
                        } else {
                          setValidationError('Failed to save. Please check your input.');
                        }
                      } else {
                        setValidationError('Network error. Please try again.');
                      }
                      return; // Keep modal open on error
                    }
                  } else if (modalMode === 'edit') {
                    const rowId = getRowId(editRowData);
                    const url = endpoint.replace(/\/$/, "");
                    const fullUrl = url.startsWith("/") ? `${backendBase}${url}/${rowId}/` : `${url}/${rowId}/`;
                    try {
                      await axios.put(fullUrl, processedValues);
                      await fetchTableData();
                      setShowModal(false);
                    } catch (err) {
                      console.error("Error updating data:", err);
                      if (axios.isAxiosError(err) && err.response) {
                        const detail = err.response.data?.detail;
                        if (detail) {
                          if (Array.isArray(detail)) {
                            const errorMessages = detail.map((e: any) => {
                              const field = e.loc ? e.loc[e.loc.length - 1] : 'field';
                              return `${field}: ${e.msg}`;
                            }).join('; ');
                            setValidationError(errorMessages);
                          } else if (typeof detail === 'string') {
                            setValidationError(detail);
                          } else {
                            setValidationError(JSON.stringify(detail));
                          }
                        } else if (err.response.data?.message) {
                          setValidationError(err.response.data.message);
                        } else {
                          setValidationError('Failed to update. Please check your input.');
                        }
                      } else {
                        setValidationError('Network error. Please try again.');
                      }
                      return; // Keep modal open on error
                    }
                  }
                }}
                style={{ display: "flex", flexDirection: "column", gap: "16px" }}
              >
                {columns.map(col => {
                  // For lookup columns, render a select dropdown
                  if (col.columnType === 'lookup' && col.entity) {
                    const endpoint = col.entity.toLowerCase();
                    const options = lookupOptions[endpoint] || [];
                    
                    // For list-type lookups (1:N or N:M), use checkbox list
                    if (col.type === 'list') {
                      const selectedValues = formValues[col.field] || [];
                      
                      return (
                        <div key={col.field} style={{ display: "flex", flexDirection: "column", gap: "8px" }}>
                          <label style={{ fontWeight: 500, color: "#334155" }}>
                            {col.label}
                            {col.required && <span style={{ color: "#ef4444", marginLeft: "4px" }}>*</span>}
                            <span style={{ fontSize: "11px", color: "#64748b", marginLeft: "6px" }}>
                              ({selectedValues.length} selected)
                            </span>
                          </label>
                          <div style={{
                            maxHeight: "160px",
                            overflowY: "auto",
                            border: "1px solid #cbd5f5",
                            borderRadius: "6px",
                            padding: "8px",
                            background: "#f8fafc",
                          }}>
                            {options.length === 0 ? (
                              <div style={{ padding: "8px", color: "#64748b", fontSize: "14px" }}>
                                No options available
                              </div>
                            ) : (
                              options.map((option: any) => {
                                const isChecked = selectedValues.includes(String(option.id));
                                return (
                                  <div 
                                    key={option.id} 
                                    style={{ 
                                      display: "flex", 
                                      alignItems: "center", 
                                      gap: "8px",
                                      padding: "6px 4px",
                                      cursor: "pointer",
                                      borderRadius: "4px",
                                    }}
                                    onMouseEnter={(e) => e.currentTarget.style.background = "#e0e7ff"}
                                    onMouseLeave={(e) => e.currentTarget.style.background = "transparent"}
                                    onClick={() => {
                                      const valueStr = String(option.id);
                                      const newSelected = isChecked
                                        ? selectedValues.filter((v: string) => v !== valueStr)
                                        : [...selectedValues, valueStr];
                                      setFormValues(fv => ({ ...fv, [col.field]: newSelected }));
                                    }}
                                  >
                                    <input
                                      type="checkbox"
                                      checked={isChecked}
                                      onChange={() => {}} // Handled by parent div onClick
                                      style={{
                                        width: "16px",
                                        height: "16px",
                                        cursor: "pointer",
                                      }}
                                    />
                                    <span style={{ fontSize: "14px", color: "#1e293b" }}>
                                      {(col.lookupField && option[col.lookupField]) || option.pages || option.stock || option.title || `ID: ${option.id}`}
                                    </span>
                                  </div>
                                );
                              })
                            )}
                          </div>
                        </div>
                      );
                    }
                    
                    // For single lookups (N:1), use regular select
                    return (
                      <div key={col.field} style={{ display: "flex", flexDirection: "column", gap: "4px" }}>
                        <label htmlFor={`modal-input-${col.field}`} style={{ fontWeight: 500, color: "#334155" }}>
                          {col.label}
                          {col.required && <span style={{ color: "#ef4444", marginLeft: "4px" }}>*</span>}
                        </label>
                        <select
                          id={`modal-input-${col.field}`}
                          value={formValues[col.field] ?? ""}
                          onChange={e => setFormValues(fv => ({ ...fv, [col.field]: e.target.value }))}
                          style={{
                            padding: "8px 12px",
                            borderRadius: "6px",
                            border: "1px solid #cbd5f5",
                            fontSize: "15px",
                            color: "#1e293b",
                            background: "#f8fafc",
                          }}
                        >
                          <option value="">-- Select {col.label} --</option>
                          {options.map((option: any) => (
                            <option key={option.id} value={option.id}>
                              {(col.lookupField && option[col.lookupField]) || option.pages || option.stock || option.title || `ID: ${option.id}`}
                            </option>
                          ))}
                        </select>
                      </div>
                    );
                  }
                  
                  // For regular fields, render input
                  // For enum fields, use dropdown
                  if (col.type === 'enum' && col.options && col.options.length > 0) {
                    return (
                      <div key={col.field} style={{ display: "flex", flexDirection: "column", gap: "4px" }}>
                        <label htmlFor={`modal-input-${col.field}`} style={{ fontWeight: 500, color: "#334155" }}>
                          {col.label}
                          {col.required && <span style={{ color: "#ef4444", marginLeft: "4px" }}>*</span>}
                        </label>
                        <select
                          id={`modal-input-${col.field}`}
                          value={formValues[col.field] ?? ""}
                          onChange={e => setFormValues(fv => ({ ...fv, [col.field]: e.target.value }))}
                          style={{
                            padding: "8px 12px",
                            borderRadius: "6px",
                            border: "1px solid #cbd5f5",
                            fontSize: "15px",
                            color: "#1e293b",
                            background: "#f8fafc",
                          }}
                        >
                          <option value="">-- Select {col.label} --</option>
                          {col.options.map((option: string) => (
                            <option key={option} value={option}>
                              {option}
                            </option>
                          ))}
                        </select>
                      </div>
                    );
                  }
                  
                  // For boolean fields, use checkbox
                  if (col.type === 'bool' || col.type === 'boolean') {
                    return (
                      <div key={col.field} style={{ display: "flex", alignItems: "center", gap: "8px" }}>
                        <input
                          id={`modal-input-${col.field}`}
                          type="checkbox"
                          checked={formValues[col.field] ?? false}
                          onChange={e => setFormValues(fv => ({ ...fv, [col.field]: e.target.checked }))}
                          style={{
                            width: "18px",
                            height: "18px",
                            cursor: "pointer",
                          }}
                        />
                        <label htmlFor={`modal-input-${col.field}`} style={{ fontWeight: 500, color: "#334155", cursor: "pointer" }}>
                          {col.label}
                          {col.required && <span style={{ color: "#ef4444", marginLeft: "4px" }}>*</span>}
                        </label>
                      </div>
                    );
                  }
                  
                  return (
                    <div key={col.field} style={{ display: "flex", flexDirection: "column", gap: "4px" }}>
                      <label htmlFor={`modal-input-${col.field}`} style={{ fontWeight: 500, color: "#334155" }}>
                        {col.label}
                        {col.required && <span style={{ color: "#ef4444", marginLeft: "4px" }}>*</span>}
                        {col.type && col.type !== 'string' && (
                          <span style={{ fontSize: "11px", color: "#64748b", marginLeft: "6px" }}>({col.type === 'list' ? 'comma-separated' : col.type})</span>
                        )}
                      </label>
                      <input
                        id={`modal-input-${col.field}`}
                        type={
                          col.type === 'int' || col.type === 'float' ? 'number' : 
                          col.type === 'date' ? 'date' : 
                          col.type === 'datetime' ? 'datetime-local' : 
                          col.type === 'time' ? 'time' : 
                          'text'
                        }
                        step={col.type === 'float' ? 'any' : undefined}
                        value={formValues[col.field] ?? ""}
                        onChange={e => setFormValues(fv => ({ ...fv, [col.field]: e.target.value }))}
                        placeholder={col.type === 'list' ? 'item1, item2, item3' : ''}
                        style={{
                          padding: "8px 12px",
                          borderRadius: "6px",
                          border: "1px solid #cbd5f5",
                          fontSize: "15px",
                          color: "#1e293b",
                          background: "#f8fafc",
                        }}
                      />
                    </div>
                  );
                })}
                <div style={{ display: "flex", justifyContent: "flex-end", gap: "12px", marginTop: "8px" }}>
                  <button
                    type="button"
                    onClick={() => setShowModal(false)}
                    style={{
                      padding: "8px 18px",
                      borderRadius: "6px",
                      border: "1px solid #cbd5f5",
                      background: "#f8fafc",
                      color: "#475569",
                      fontWeight: 500,
                      cursor: "pointer",
                    }}
                  >Cancel</button>
                  <button
                    type="submit"
                    style={{
                      padding: "8px 18px",
                      borderRadius: "6px",
                      border: "none",
                      background: "linear-gradient(90deg, #2563eb 0%, #1e40af 100%)",
                      color: "#fff",
                      fontWeight: 700,
                      cursor: "pointer",
                    }}
                  >Save</button>
                </div>
              </form>
              <button
                type="button"
                onClick={() => setShowModal(false)}
                style={{
                  position: "absolute",
                  top: "12px",
                  right: "12px",
                  background: "none",
                  border: "none",
                  fontSize: "20px",
                  color: "#64748b",
                  cursor: "pointer",
                }}
                aria-label="Close"
              >×</button>
            </div>
          </div>,
          document.body
        )
      )}

      {normalizedRows.length === 0 || columns.length === 0 ? (
        <div
          style={{
            padding: "24px",
            textAlign: "center",
            color: "#64748b",
            border: "1px dashed #cbd5f5",
            borderRadius: "8px",
            backgroundColor: "#f8fafc",
          }}
        >
          No data available for{" "}
          {dataBinding?.entity ? `${dataBinding.entity}` : "this table"}.
        </div>
      ) : (
        <div className="table-scroll">
          <table style={tableStyle}>
            {resolvedOptions.showHeader && (
              <thead>
                <tr style={{ backgroundColor: "#1e293b", color: "#ffffff" }}>
                  {columns.map((column) => (
                    <th
                      key={`${id}-${column.field}`}
                      style={{
                        textAlign: "left",
                        padding: "10px 12px",
                        fontWeight: 600,
                        letterSpacing: "0.01em",
                      }}
                    >
                      <div style={{ display: "flex", alignItems: "center", gap: "4px" }}>
                        <span>{column.label}</span>
                        <ColumnSort
                          column={column}
                          currentSort={sortConfig}
                          onSortChange={handleSortChange}
                        />
                        <ColumnFilter
                          column={column}
                          onFilterChange={handleFilterChange}
                          currentValue={columnFilters[column.field] || ""}
                        />
                      </div>
                    </th>
                  ))}
                  {resolvedOptions.actionButtons && (
                    <th style={{ textAlign: "center", padding: "10px 4px", fontWeight: 600, width: "40px", minWidth: "40px", maxWidth: "40px", overflow: "hidden" }}></th>
                  )}
                </tr>
              </thead>
            )}
            <tbody>
              {visibleRows.map((row, rowIndex) => {
                const actualRowIndex = (currentPage - 1) * pageSize + rowIndex;
                const isSelected = selectedRowIndex === actualRowIndex;
                
                return (
                <tr
                  key={`${id}-row-${rowIndex}`}
                  onClick={() => {
                    setSelectedRowIndex(actualRowIndex);
                    setSelectedRow(id, row);
                    console.log(`[TableComponent] Row selected in table ${id}:`, row);
                  }}
                  style={{
                    backgroundColor: isSelected
                      ? "#dbeafe"
                      : resolvedOptions.stripedRows && rowIndex % 2 === 1
                        ? "#f8fafc"
                        : "#ffffff",
                    borderBottom: "1px solid #e2e8f0",
                    cursor: "pointer",
                    transition: "background-color 0.2s",
                  }}
                  onMouseEnter={(e) => {
                    if (!isSelected) {
                      e.currentTarget.style.backgroundColor = "#f1f5f9";
                    }
                  }}
                  onMouseLeave={(e) => {
                    if (!isSelected) {
                      e.currentTarget.style.backgroundColor = 
                        resolvedOptions.stripedRows && rowIndex % 2 === 1
                          ? "#f8fafc"
                          : "#ffffff";
                    }
                  }}
                >
                  {columns.map((column) => {
                    // For lookup columns, get nested value using the API relationship key
                    let cellValue;
                    if (column.columnType === 'lookup' && column.lookupField) {
                      // Check if this is a list type (1:N or N:M relationship)
                      if (column.type === 'list') {
                        // Get the array of related objects using the field name
                        const relatedArray = row[column.field];
                        if (Array.isArray(relatedArray) && column.lookupField) {
                          // Extract the field values from each object and join them
                          cellValue = relatedArray
                            .map(item => item[column.lookupField!])
                            .filter(val => val !== null && val !== undefined)
                            .join(', ');
                        } else {
                          cellValue = '';
                        }
                      } else {
                        // Single value lookup (N:1 relationship)
                        // Try the relationship key first (e.g., customer.name)
                        if (column.relationshipKey) {
                          const nestedPath = `${column.relationshipKey}.${column.lookupField}`;
                          cellValue = getNestedValue(row, nestedPath);
                        }
                        // Fallback to direct path (e.g., customer.name)
                        if (!cellValue) {
                          const nestedPath = `${column.field}.${column.lookupField}`;
                          cellValue = getNestedValue(row, nestedPath);
                        }
                      }
                    } else {
                      cellValue = row[column.field];
                    }
                    
                    return (
                      <td
                        key={`${id}-row-${rowIndex}-cell-${column.field}`}
                        style={{
                          padding: "10px 12px",
                          color: "#1f2937",
                          whiteSpace: "nowrap",
                          textOverflow: "ellipsis",
                          overflow: "hidden",
                          maxWidth: "200px",
                        }}
                        title={formatCellValue(cellValue)}
                      >
                        {formatCellValue(cellValue)}
                      </td>
                    );
                  })}
                  {resolvedOptions.actionButtons && (
                    <td style={{ textAlign: "center", padding: "10px 2px", width: "40px", minWidth: "40px", maxWidth: "40px", overflow: "hidden" }}>
                      <div style={{ display: "flex", flexDirection: "row", justifyContent: "center", alignItems: "center", gap: "2px", width: "100%" }}>
                        <button
                          style={{
                            background: "none",
                            border: "none",
                            padding: "2px",
                            cursor: "pointer",
                            display: "flex",
                            alignItems: "center",
                          }}
                          type="button"
                          title="Edit"
                          onClick={() => {
                            setModalMode('edit');
                            setEditRowData(row);
                            setShowModal(true);
                          }}
                        >
                          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#f59e42" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M12 20h9"/><path d="M16.5 3.5a2.121 2.121 0 0 1 3 3L7 19l-4 1 1-4 12.5-12.5z"/></svg>
                        </button>
                        <button
                          style={{
                            background: "none",
                            border: "none",
                            padding: "2px",
                            cursor: "pointer",
                            display: "flex",
                            alignItems: "center",
                          }}
                          type="button"
                          title="Remove"
                          onClick={async () => {
                            const endpoint = getEndpoint();
                            const rowId = getRowId(row);
                            const url = endpoint.replace(/\/$/, "");
                            const backendBase = process.env.REACT_APP_API_URL || "http://localhost:8000";
                            const fullUrl = url.startsWith("/") ? `${backendBase}${url}/${rowId}/` : `${url}/${rowId}/`;
                            try {
                              await axios.delete(fullUrl);
                              await fetchTableData();
                            } catch (err) {
                              console.error("Error deleting data:", err);
                            }
                          }}
                        >
                          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#ef4444" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><polyline points="3 6 5 6 21 6"/><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m5 0V4a2 2 0 0 1 2-2h2a2 2 0 0 1 2 2v2"/><line x1="10" y1="11" x2="10" y2="17"/><line x1="14" y1="11" x2="14" y2="17"/></svg>
                        </button>
                      </div>
                    </td>
                  )}
                </tr>
              );
              })}
            </tbody>
          </table>
        </div>
      )}

      {resolvedOptions.showPagination && filteredRows.length > 0 && (
        <div
          style={{
            display: "flex",
            alignItems: "center",
            justifyContent: "space-between",
            gap: "12px",
            fontSize: "13px",
            color: "#475569",
          }}
        >
          <span>
            Showing {visibleRows.length} of {sortedAndFilteredRows.length} rows
            {Object.keys(columnFilters).length > 0 && (
              <span style={{ color: "#2563eb", marginLeft: "4px" }}>
                (filtered from {normalizedRows.length})
              </span>
            )}
          </span>
          <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
            <button
              type="button"
              onClick={() => setCurrentPage((prev) => Math.max(1, prev - 1))}
              disabled={currentPage === 1}
              style={{
                padding: "6px 12px",
                borderRadius: "6px",
                border: "1px solid #cbd5f5",
                backgroundColor: currentPage === 1 ? "#e2e8f0" : "#ffffff",
                cursor: currentPage === 1 ? "not-allowed" : "pointer",
              }}
            >
              Prev
            </button>
            <span>
              Page {currentPage} of {totalPages}
            </span>
            <button
              type="button"
              onClick={() =>
                setCurrentPage((prev) => Math.min(totalPages, prev + 1))
              }
              disabled={currentPage === totalPages}
              style={{
                padding: "6px 12px",
                borderRadius: "6px",
                border: "1px solid #cbd5f5",
                backgroundColor:
                  currentPage === totalPages ? "#e2e8f0" : "#ffffff",
                cursor:
                  currentPage === totalPages ? "not-allowed" : "pointer",
              }}
            >
              Next
            </button>
          </div>
        </div>
      )}
    </div>
  );
};