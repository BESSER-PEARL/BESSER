// A readable label for an identifier: "unit_price" -> "Unit price", "orderDate" -> "Order date";
// an acronym ("customerID" -> "Customer ID") and words already spaced keep their capitals.
export const humanize = (value: string): string => {
  if (!value) {
    return "";
  }
  return value
    .replace(/[_-]+/g, " ")
    .replace(/([a-z])([A-Z])(?=[a-z])/g, (_, before, capital) => `${before} ${capital.toLowerCase()}`)
    .replace(/([a-z])([A-Z])/g, "$1 $2")
    .replace(/\s+/g, " ")
    .trim()
    .replace(/^./, (str) => str.toUpperCase());
};
