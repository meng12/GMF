package happy.coding.io.table;

import javax.swing.table.TableModel;

public abstract class SeparatorPolicy {

	protected TableModel tableModel;

	public SeparatorPolicy() {
	}

	public SeparatorPolicy(TableModel tableModel) {
		this.tableModel = tableModel;
	}

	public TableModel getTableModel() {
		return tableModel;
	}

	public void setTableModel(TableModel tableModel) {
		this.tableModel = tableModel;
	}

	abstract boolean hasSeparatorAt(int row);
}
