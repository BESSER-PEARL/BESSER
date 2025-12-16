import React, { createContext, useContext, useState } from "react";

interface TableContextValue {
  selectedRows: Record<string, any>;
  setSelectedRow: (tableId: string, row: any) => void;
  getSelectedRow: (tableId: string) => any;
  refreshTable: (tableId: string) => void;
  registerTableRefresh: (tableId: string, refreshFn: () => void) => void;
}

const TableContext = createContext<TableContextValue | undefined>(undefined);

export const TableProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [selectedRows, setSelectedRows] = useState<Record<string, any>>({});
  const [tableRefreshFunctions, setTableRefreshFunctions] = useState<Record<string, () => void>>({});

  const setSelectedRow = (tableId: string, row: any) => {
    setSelectedRows((prev) => ({
      ...prev,
      [tableId]: row,
    }));
  };

  const getSelectedRow = (tableId: string) => {
    return selectedRows[tableId];
  };

  const registerTableRefresh = (tableId: string, refreshFn: () => void) => {
    setTableRefreshFunctions((prev) => ({
      ...prev,
      [tableId]: refreshFn,
    }));
  };

  const refreshTable = (tableId: string) => {
    const refreshFn = tableRefreshFunctions[tableId];
    if (refreshFn) {
      console.log(`[TableContext] Refreshing table: ${tableId}`);
      refreshFn();
    } else {
      console.warn(`[TableContext] No refresh function registered for table: ${tableId}`);
    }
  };

  return (
    <TableContext.Provider value={{ selectedRows, setSelectedRow, getSelectedRow, refreshTable, registerTableRefresh }}>
      {children}
    </TableContext.Provider>
  );
};

export const useTableContext = () => {
  const context = useContext(TableContext);
  if (!context) {
    throw new Error("useTableContext must be used within a TableProvider");
  }
  return context;
};