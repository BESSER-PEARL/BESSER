import React, { useState } from "react";
import axios from "axios";
import { useTableContext } from "../contexts/TableContext";

export interface MethodParameter {
  name: string;
  type: string;
  required?: boolean;
  default?: any;
}

export interface MethodButtonProps {
  // New props for JSON-based configuration
  endpoint?: string;
  isInstanceMethod?: boolean;
  instanceSourceTableId?: string;
  
  // Legacy props (kept for backwards compatibility)
  entityType?: string;
  entityId?: number | string;
  methodName?: string;
  
  // Common props
  label?: string;
  parameters?: MethodParameter[];
  isClassMethod?: boolean;
  onSuccess?: (result: any) => void;
  onError?: (error: any) => void;
  className?: string;
  style?: React.CSSProperties;
  backendUrl?: string;
}

/**
 * MethodButton component for executing backend class/instance methods
 * 
 * Usage examples:
 * 
 * // Instance method without parameters
 * <MethodButton
 *   entityType="library"
 *   entityId={1}
 *   methodName="has_address"
 *   label="Check Address"
 * />
 * 
 * // Instance method with parameters
 * <MethodButton
 *   entityType="book"
 *   entityId={5}
 *   methodName="is_old"
 *   label="Check if Old"
 *   parameters={[{ name: "years", type: "number", required: false, default: 10 }]}
 * />
 * 
 * // Class method (static) with parameters
 * <MethodButton
 *   entityType="book"
 *   methodName="count_long_books"
 *   label="Count Long Books"
 *   isClassMethod={true}
 *   parameters={[{ name: "min_pages", type: "number", required: false, default: 300 }]}
 * />
 */
export const MethodButton: React.FC<MethodButtonProps> = ({
  endpoint: endpointProp,
  isInstanceMethod = false,
  instanceSourceTableId,
  entityType,
  entityId,
  methodName,
  label,
  parameters = [],
  isClassMethod = false,
  onSuccess,
  onError,
  className = "",
  style = {},
  backendUrl = process.env.REACT_APP_API_URL || "http://localhost:8000",
}) => {
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);
  const [showModal, setShowModal] = useState(false);
  const [showSuccessToast, setShowSuccessToast] = useState(false);
  const [showErrorToast, setShowErrorToast] = useState(false);
  const [paramValues, setParamValues] = useState<Record<string, any>>(() => {
    // Initialize with default values
    const defaults: Record<string, any> = {};
    parameters.forEach(param => {
      if (param.default !== undefined) {
        defaults[param.name] = param.default;
      }
    });
    return defaults;
  });

  // Access table context to get selected row
  const tableContext = useTableContext();

  const handleExecute = async () => {
    setLoading(true);
    setError(null);
    setResult(null);

    try {
      // Build the endpoint URL
      let finalEndpoint: string;
      
      if (endpointProp) {
        // New JSON-based configuration
        finalEndpoint = endpointProp;
        
        // If this is an instance method, we need to replace {book_id} or similar placeholders
        if (isInstanceMethod && instanceSourceTableId) {
          const selectedRow = tableContext.getSelectedRow(instanceSourceTableId);
          
          if (!selectedRow) {
            throw new Error(`Please select a row in the table first`);
          }
          
          // Get the ID from the selected row
          const rowId = selectedRow.id;
          if (!rowId) {
            throw new Error("Selected row does not have an ID");
          }
          
          // Replace placeholders like {book_id}, {entity_id}, etc.
          finalEndpoint = finalEndpoint.replace(/\{[^}]+\}/g, String(rowId));
          
          console.log(`[MethodButton] Using selected row ID: ${rowId}`);
        }
        
        // Ensure endpoint starts with /
        if (!finalEndpoint.startsWith('/')) {
          finalEndpoint = '/' + finalEndpoint;
        }
        
        // Prepend backend URL
        finalEndpoint = backendUrl + finalEndpoint;
      } else {
        // Legacy method_config based configuration
        if (isClassMethod) {
          // Class method: /entity/methods/method_name/
          finalEndpoint = `${backendUrl}/${entityType}/methods/${methodName}/`;
        } else {
          // Instance method: /entity/{id}/methods/method_name/
          if (!entityId) {
            throw new Error("Entity ID is required for instance methods");
          }
          finalEndpoint = `${backendUrl}/${entityType}/${entityId}/methods/${methodName}/`;
        }
      }

      // Prepare request body with parameters
      // Wrap parameters in a 'params' object to match FastAPI Body(embed=True) expectation
      const requestBody = Object.keys(paramValues).length > 0 ? { params: paramValues } : {};

      console.log(`[MethodButton] Executing method:`, {
        endpoint: finalEndpoint,
        requestBody,
        isInstanceMethod,
        instanceSourceTableId,
      });

      // Execute the method via POST request
      const response = await axios.post(finalEndpoint, requestBody, {
        headers: {
          'Content-Type': 'application/json',
        },
      });

      console.log(`[MethodButton] Method executed successfully:`, response.data);
      setResult(response.data);
      
      // Refresh the source table if this is an instance method
      if (isInstanceMethod && instanceSourceTableId) {
        console.log(`[MethodButton] Refreshing table: ${instanceSourceTableId}`);
        tableContext.refreshTable(instanceSourceTableId);
      }
      
      if (onSuccess) {
        onSuccess(response.data);
      }

      // Close modal and show success toast
      if (parameters.length > 0) {
        setShowModal(false);
      }
      
      // Show brief success confirmation
      setShowSuccessToast(true);
      setTimeout(() => setShowSuccessToast(false), 3000);
    } catch (err: any) {
      console.error(`[MethodButton] Error executing method:`, err);
      
      // Handle different error response formats
      let errorMessage = "Error executing method";
      const statusCode = err.response?.status;
      
      if (err.response?.data) {
        const data = err.response.data;
        
        // Try to extract error message from various formats
        if (data.detail) {
          const detail = data.detail;
          
          // If detail is an array (FastAPI validation errors - 422)
          if (Array.isArray(detail)) {
            errorMessage = detail.map((e: any) => {
              const field = e.loc ? e.loc[e.loc.length - 1] : 'field';
              return `${field}: ${e.msg}`;
            }).join('; ');
          } 
          // If detail is an object
          else if (typeof detail === 'object') {
            errorMessage = detail.message || detail.error || JSON.stringify(detail);
          }
          // If detail is a string
          else {
            errorMessage = String(detail);
          }
        } else if (data.message) {
          errorMessage = data.message;
        } else if (data.error) {
          errorMessage = data.error;
        } else if (typeof data === 'string') {
          errorMessage = data;
        } else {
          // For 500 errors, try to show something useful
          errorMessage = `Server error (${statusCode || 'unknown'}): ${JSON.stringify(data).substring(0, 200)}`;
        }
      } else if (err.message) {
        // Network errors or other axios errors
        if (statusCode) {
          errorMessage = `HTTP ${statusCode}: ${err.message}`;
        } else {
          errorMessage = err.message;
        }
      }
      
      // Add status code prefix for server errors if not already included
      if (statusCode && statusCode >= 500 && !errorMessage.includes(String(statusCode))) {
        errorMessage = `Server Error (${statusCode}): ${errorMessage}`;
      }
      
      setError(errorMessage);
      
      // Keep modal open on error so user can retry
      // Show error toast only if no modal (for methods without parameters)
      if (parameters.length === 0) {
        setShowErrorToast(true);
        setTimeout(() => setShowErrorToast(false), 5000);
      }
      
      if (onError) {
        onError(err);
      }
    } finally {
      setLoading(false);
    }
  };

  const handleClick = () => {
    // Reset error and result state when opening modal or executing
    setError(null);
    setResult(null);
    
    // If method has parameters, show modal for input
    if (parameters.length > 0) {
      setShowModal(true);
    } else {
      // Execute directly if no parameters
      handleExecute();
    }
  };

  const handleParamChange = (paramName: string, value: any) => {
    setParamValues(prev => ({
      ...prev,
      [paramName]: value,
    }));
  };

  const renderParameterInput = (param: MethodParameter) => {
    const value = paramValues[param.name] ?? param.default ?? "";

    switch (param.type.toLowerCase()) {
      case "number":
      case "int":
      case "integer":
        return (
          <input
            type="number"
            step="1"
            value={value}
            onChange={(e) => handleParamChange(param.name, parseInt(e.target.value) || 0)}
            required={param.required}
            style={{ 
              width: "100%", 
              padding: "8px 12px",
              borderRadius: "6px",
              border: "1px solid #cbd5f5",
              fontSize: "15px",
              color: "#1e293b",
              background: "#f8fafc",
              boxSizing: "border-box",
            }}
          />
        );
      case "float":
      case "double":
      case "decimal":
        return (
          <input
            type="number"
            step="any"
            value={value}
            onChange={(e) => handleParamChange(param.name, parseFloat(e.target.value) || 0)}
            required={param.required}
            style={{ 
              width: "100%", 
              padding: "8px 12px",
              borderRadius: "6px",
              border: "1px solid #cbd5f5",
              fontSize: "15px",
              color: "#1e293b",
              background: "#f8fafc",
              boxSizing: "border-box",
            }}
          />
        );
      case "boolean":
      case "bool":
        return (
          <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
            <input
              type="checkbox"
              checked={!!value}
              onChange={(e) => handleParamChange(param.name, e.target.checked)}
              style={{ width: "18px", height: "18px", cursor: "pointer" }}
            />
            <span style={{ fontSize: "14px", color: "#64748b" }}>
              {value ? "True" : "False"}
            </span>
          </div>
        );
      case "string":
      case "str":
      case "text":
      default:
        return (
          <input
            type="text"
            value={value}
            onChange={(e) => handleParamChange(param.name, e.target.value)}
            required={param.required}
            style={{ 
              width: "100%", 
              padding: "8px 12px",
              borderRadius: "6px",
              border: "1px solid #cbd5f5",
              fontSize: "15px",
              color: "#1e293b",
              background: "#f8fafc",
              boxSizing: "border-box",
            }}
          />
        );
    }
  };

  return (
    <>
      <button
        onClick={handleClick}
        disabled={loading}
        className={className}
        style={{
          padding: "8px 16px",
          backgroundColor: loading ? "#ccc" : "#007bff",
          color: "white",
          border: "none",
          borderRadius: "4px",
          cursor: loading ? "not-allowed" : "pointer",
          ...style,
        }}
      >
        {loading ? "Executing..." : label || methodName}
      </button>

      {/* Success toast */}
      {showSuccessToast && (
        <div style={{
          position: "fixed",
          bottom: "20px",
          right: "20px",
          padding: "12px 20px",
          backgroundColor: "#10b981",
          color: "white",
          borderRadius: "8px",
          boxShadow: "0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05)",
          zIndex: 2000,
          fontWeight: 500,
          fontSize: "14px",
          display: "flex",
          alignItems: "center",
          gap: "8px",
        }}>
          <span style={{ fontSize: "18px" }}>✓</span>
          Operation completed successfully
        </div>
      )}

      {/* Error toast */}
      {showErrorToast && error && (
        <div style={{
          position: "fixed",
          bottom: "20px",
          right: "20px",
          padding: "12px 20px",
          backgroundColor: "#ef4444",
          color: "white",
          borderRadius: "8px",
          boxShadow: "0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05)",
          zIndex: 2000,
          fontWeight: 500,
          fontSize: "14px",
          maxWidth: "400px",
          wordBreak: "break-word",
        }}>
          <strong style={{ display: "block", marginBottom: "4px" }}>Error:</strong>
          {error}
        </div>
      )}

      {/* Parameter input modal */}
      {showModal && (
        <div style={{
          position: "fixed",
          top: 0,
          left: 0,
          right: 0,
          bottom: 0,
          backgroundColor: "rgba(0, 0, 0, 0.5)",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          zIndex: 1000,
        }}>
          <div style={{
            backgroundColor: "white",
            padding: "24px",
            borderRadius: "12px",
            minWidth: "400px",
            maxWidth: "500px",
            boxShadow: "0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04)",
            position: "relative",
          }}>
            <h3 style={{ 
              marginTop: 0,
              marginBottom: "20px",
              fontSize: "18px",
              fontWeight: 700,
              color: "#1e293b",
            }}>
              {label} - Parameters
            </h3>

            {/* Error display inside modal */}
            {error && (
              <div style={{
                padding: "12px",
                marginBottom: "16px",
                backgroundColor: "#fee2e2",
                border: "1px solid #fecaca",
                borderRadius: "6px",
                color: "#991b1b",
                fontSize: "14px",
              }}>
                {error}
              </div>
            )}

            <form onSubmit={(e) => {
              e.preventDefault();
              handleExecute();
            }}>
              {parameters.map(param => (
                <div key={param.name} style={{ marginBottom: "15px" }}>
                  <label style={{ 
                    display: "block", 
                    marginBottom: "5px", 
                    fontWeight: 600,
                    color: "#334155",
                    fontSize: "14px"
                  }}>
                    {param.name}
                    {param.required && <span style={{ color: "#ef4444", marginLeft: "4px" }}>*</span>}
                    {param.type && (
                      <span style={{ fontSize: "11px", color: "#64748b", marginLeft: "6px", fontWeight: 400 }}>
                        ({param.type})
                      </span>
                    )}
                    {param.default !== undefined && (
                      <span style={{ fontSize: "11px", color: "#64748b", marginLeft: "6px", fontWeight: 400 }}>
                        {" "}default: {String(param.default)}
                      </span>
                    )}
                  </label>
                  {renderParameterInput(param)}
                </div>
              ))}

              <div style={{ display: "flex", gap: "12px", justifyContent: "flex-end", marginTop: "24px" }}>
                <button
                  type="button"
                  onClick={() => setShowModal(false)}
                  style={{
                    padding: "8px 18px",
                    backgroundColor: "#f8fafc",
                    color: "#475569",
                    border: "1px solid #cbd5f5",
                    borderRadius: "6px",
                    cursor: "pointer",
                    fontWeight: 500,
                    fontSize: "14px",
                  }}
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  disabled={loading}
                  style={{
                    padding: "8px 18px",
                    background: loading ? "#cbd5e1" : "linear-gradient(90deg, #2563eb 0%, #1e40af 100%)",
                    color: "white",
                    border: "none",
                    borderRadius: "6px",
                    cursor: loading ? "not-allowed" : "pointer",
                    fontWeight: 600,
                    fontSize: "14px",
                  }}
                >
                  {loading ? "Executing..." : "Execute"}
                </button>
              </div>
            </form>
            <button
              type="button"
              onClick={() => setShowModal(false)}
              style={{
                position: "absolute",
                top: "16px",
                right: "16px",
                background: "none",
                border: "none",
                fontSize: "24px",
                color: "#64748b",
                cursor: "pointer",
                lineHeight: 1,
                padding: "4px",
              }}
              aria-label="Close"
            >×</button>
          </div>
        </div>
      )}
    </>
  );
};