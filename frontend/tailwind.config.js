/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        brand: { 50: "#eef7fb", 100: "#d5ecf5", 200: "#acd8ea", 300: "#77bcd9", 400: "#3f9ac3", 500: "#1f7ea9", 600: "#17668c", 700: "#155372", 800: "#16465e", 900: "#163b50" },
        teal: { 50: "#ecfbf8", 100: "#cff5ec", 500: "#13a38b", 600: "#0d8572", 700: "#0e6a5c" },
        ink: { 900: "#14213a", 700: "#3a4760", 500: "#66728a", 300: "#a9b2c3", 200: "#d6dce6", 100: "#eef1f6", 50: "#f7f9fc" },
      },
      fontFamily: {
        sans: ['"Inter"', '"Segoe UI"', "system-ui", "-apple-system", "Roboto", "sans-serif"],
        serif: ['"Source Serif 4"', "Georgia", "Cambria", "serif"],
      },
      boxShadow: { card: "0 1px 2px rgba(20,33,58,.05), 0 1px 1px rgba(20,33,58,.03)" },
    },
  },
  plugins: [],
};
