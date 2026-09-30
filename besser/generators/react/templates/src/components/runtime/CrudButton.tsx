import React, { CSSProperties, useState } from "react";
import { useNavigate } from "react-router-dom";
import { TableActionName, useTableContext } from "../../contexts/TableContext";

export interface CrudButtonProps {
  id?: string;
  label?: string;
  action: TableActionName;
  // The table bound to the entity: its dialog, endpoint and selected row are used
  tableId: string;
  entity?: string;
  // Route of the table's page when it is not on this one (create only)
  targetPath?: string;
  confirmMessage?: string;
  className?: string;
  style?: CSSProperties;
}

/**
 * Create / update / delete button of a GUI model. It drives the bound table:
 * create opens the table's Add dialog, update opens its Edit dialog on the
 * selected row, delete removes the selected row - the same requests the
 * table's own controls send.
 */
export const CrudButton: React.FC<CrudButtonProps> = ({
  id,
  label,
  action,
  tableId,
  entity,
  targetPath,
  confirmMessage,
  className,
  style,
}) => {
  const { runTableAction } = useTableContext();
  const navigate = useNavigate();
  const [message, setMessage] = useState<string | null>(null);

  const handleClick = () => {
    setMessage(null);
    if (confirmMessage && !window.confirm(confirmMessage)) {
      return;
    }
    if (targetPath) {
      runTableAction(tableId, action, true);
      navigate(targetPath);
      return;
    }
    const problem = runTableAction(tableId, action);
    if (problem) {
      setMessage(entity && action !== "create" ? problem.replace("a row", `a ${entity}`) : problem);
    }
  };

  return (
    <>
      <button id={id} type="button" className={className} style={style} onClick={handleClick}>
        {label || `${action.charAt(0).toUpperCase()}${action.slice(1)}${entity ? ` ${entity}` : ""}`}
      </button>
      {message && (
        <span role="alert" style={{ marginLeft: "8px", color: "#b91c1c", fontSize: "13px" }}>
          {message}
        </span>
      )}
    </>
  );
};
