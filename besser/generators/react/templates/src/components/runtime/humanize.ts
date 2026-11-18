// A readable label for an identifier: "unit_price" -> "Unit price", "orderDate" -> "Order Date".
export const humanize = (value: string): string => {
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
