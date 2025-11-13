/** @type {import('tailwindcss').Config} */
module.exports = {
  darkMode: "class", // enable dark mode via class
  content: [
    "./app/**/*.{ts,tsx,js,jsx}",
    "./components/**/*.{ts,tsx,js,jsx}",
  ],
  theme: {
    extend: {
      colors: {
        background: "var(--background)",
        foreground: "var(--foreground)",
        card: "var(--card)",
        cardForeground: "var(--card-foreground)",
        primary: "var(--primary)",
        primaryForeground: "var(--primary-foreground)",
        secondary: "var(--secondary)",
        secondaryForeground: "var(--secondary-foreground)",
        muted: "var(--muted)",
        mutedForeground: "var(--muted-foreground)",
        accent: "var(--accent)",
        accentForeground: "var(--accent-foreground)",
        destructive: "var(--destructive)",
        destructiveForeground: "var(--destructive-foreground)",
        border: "var(--border)",
        input: "var(--input)",
        ring: "var(--ring)",
      },
      borderRadius: {
        sm: "0.25rem",
        md: "0.375rem",
        lg: "0.5rem",
        xl: "0.75rem",
      },
    },
  },
  plugins: [],
};
