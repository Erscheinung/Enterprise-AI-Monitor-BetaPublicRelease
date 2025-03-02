.App {
  background-color: #f7f9fc;
  min-height: 100vh;
  padding: 20px 0;
}

.MuiCard-root {
  box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
  transition: all 0.3s ease;
}

.MuiCard-root:hover {
  transform: translateY(-4px);
  box-shadow: 0 8px 16px rgba(0, 0, 0, 0.1);
}

.MuiCardHeader-root {
  background-color: #f5f8ff;
  border-bottom: 1px solid #eaeaea;
}

/* Status colors */
.status-green {
  color: #388e3c;
  font-weight: bold;
}

.status-yellow {
  color: #f57c00;
  font-weight: bold;
}

.status-red {
  color: #d32f2f;
  font-weight: bold;
}

/* Table styling */
.MuiTableCell-head {
  background-color: #f5f8ff;
  font-weight: bold;
}

.MuiTableRow-root:nth-of-type(even) {
  background-color: #fafafa;
}

/* Status Paper styling */
.status-paper {
  padding: 16px;
  border-radius: 8px;
  margin-bottom: 16px;
  transition: all 0.3s ease;
}

.status-paper:hover {
  transform: scale(1.02);
}

/* Responsive adjustments */
@media (max-width: 960px) {
  .responsive-container {
    flex-direction: column;
  }
}