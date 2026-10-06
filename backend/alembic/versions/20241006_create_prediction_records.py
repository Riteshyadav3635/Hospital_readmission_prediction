"""Create prediction records table.

Revision ID: 20241006_pred_records
Revises:
Create Date: 2024-10-06 00:00:00.000000
"""

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = "20241006_pred_records"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "prediction_records",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("patient_reference", sa.String(length=255), nullable=True),
        sa.Column("input_features", sa.JSON(), nullable=False),
        sa.Column("prediction", sa.Integer(), nullable=False),
        sa.Column("probability", sa.Float(), nullable=False),
        sa.Column("risk_level", sa.String(length=64), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_prediction_records_id"), "prediction_records", ["id"], unique=False)


def downgrade() -> None:
    op.drop_index(op.f("ix_prediction_records_id"), table_name="prediction_records")
    op.drop_table("prediction_records")
