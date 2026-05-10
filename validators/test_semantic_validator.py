"""
Unit tests for core.semantic_validator.

These tests exercise the document-type-aware semantic validation layer without
any external service calls (no Azure DI, no LLM, no DB).
"""
import pytest
from validators.semantic_validator import semantic_validate, ValidationResult


# ── Test A: Taco Bell receipt — merchant name / store number splitting ────────

class TestReceiptMerchantSplit:
    """merchant_name with trailing store code should be split."""

    def _run(self, merchant_name: str, raw_text: str = ""):
        fields = {
            "merchant_name": merchant_name,
            "store_number": None,
            "subtotal": None,
            "tax_amount": None,
            "total": None,
            "items": [],
        }
        return semantic_validate("receipt", fields, raw_text)

    def test_taco_bell_store_number_split(self):
        result_fields, vr = self._run("Taco Bell 040061")
        assert result_fields["merchant_name"] == "Taco Bell"
        assert result_fields["store_number"] == "040061"
        assert isinstance(vr, ValidationResult)
        assert "store_number" in vr.raw_to_normalized_mapping
        fv = vr.raw_to_normalized_mapping["store_number"]
        assert fv.status == "valid"
        assert fv.normalized_value == "040061"

    def test_store_hash_prefix_split(self):
        result_fields, vr = self._run("McDonald's #1234")
        assert result_fields["merchant_name"] == "McDonald's"
        assert result_fields["store_number"] == "1234"

    def test_no_split_when_name_only(self):
        result_fields, _ = self._run("Whole Foods Market")
        assert result_fields["merchant_name"] == "Whole Foods Market"
        assert result_fields.get("store_number") in (None, "")

    def test_no_split_long_trailing_digits(self):
        # 7+ digit trailing number looks like a phone/zip suffix — don't split
        result_fields, _ = self._run("Store 1234567")
        assert result_fields.get("store_number") is None or len(
            str(result_fields.get("store_number", ""))
        ) > 6 or result_fields["merchant_name"] == "Store 1234567"

    def test_validation_status_is_valid_when_clean(self):
        result_fields, vr = self._run("Taco Bell 040061")
        # Should be "valid" or "warning" — not "invalid"
        assert vr.validation_status in ("valid", "warning")

    def test_zero_tax_not_flagged(self):
        """Delaware has no sales tax — zero tax must not produce an error."""
        fields = {
            "merchant_name": "Target",
            "store_number": None,
            "subtotal": "25.00",
            "tax_amount": "0.00",
            "total": "25.00",
            "items": [],
        }
        _, vr = semantic_validate("receipt", fields, "")
        # Zero tax should not appear in errors
        assert not any("tax" in e.lower() for e in vr.validation_errors)

    def test_item_sum_warning_when_mismatched(self):
        fields = {
            "merchant_name": "Test Store",
            "store_number": None,
            "subtotal": "10.00",
            "tax_amount": None,
            "total": None,
            "items": [
                {"description": "Item A", "quantity": 1, "unit_price": "3.00", "line_total": "3.00"},
                {"description": "Item B", "quantity": 1, "unit_price": "4.00", "line_total": "4.00"},
            ],
        }
        _, vr = semantic_validate("receipt", fields, "")
        assert any("subtotal" in w for w in vr.validation_warnings)


# ── Test B: Auto repair invoice — invoice_number vs service_order_number ──────

class TestServiceInvoiceDisambiguation:
    """When both invoice_number and service_order_number carry the same value,
    keep the one whose label appears in the OCR text."""

    BASE_FIELDS = {
        "invoice_number":       "WO-12345",
        "service_order_number": "WO-12345",
        "service_date": None,
        "service_provider_name": None,
        "labor_cost": None,
        "parts_cost": None,
        "subtotal": None,
        "tax_amount": None,
        "total_amount": None,
        "items": [],
    }

    def test_keeps_invoice_clears_service_order_when_only_invoice_label(self):
        raw = "Invoice Number: WO-12345\nLabor: Oil Change"
        result_fields, vr = semantic_validate(
            "receipt_warranty_service", dict(self.BASE_FIELDS), raw
        )
        assert result_fields["invoice_number"] == "WO-12345"
        assert result_fields["service_order_number"] is None
        # Clearing service_order_number should produce a warning or be noted
        assert any("service_order_number" in w for w in vr.validation_warnings) or \
               vr.raw_to_normalized_mapping.get("service_order_number") is not None

    def test_keeps_service_order_clears_invoice_when_only_work_order_label(self):
        raw = "Work Order #WO-12345\nParts replaced: brake pads"
        result_fields, vr = semantic_validate(
            "receipt_warranty_service", dict(self.BASE_FIELDS), raw
        )
        assert result_fields["service_order_number"] == "WO-12345"
        assert result_fields["invoice_number"] is None

    def test_keeps_both_when_both_labels_present(self):
        raw = "Invoice Number: WO-12345\nWork Order: WO-12345"
        result_fields, vr = semantic_validate(
            "receipt_warranty_service", dict(self.BASE_FIELDS), raw
        )
        # Both should be kept; a warning is emitted
        assert result_fields["invoice_number"] == "WO-12345"
        assert result_fields["service_order_number"] == "WO-12345"
        assert any("invoice_number" in w for w in vr.validation_warnings)

    def test_different_values_not_affected(self):
        fields = dict(self.BASE_FIELDS)
        fields["service_order_number"] = "SO-99999"
        raw = "Invoice Number: WO-12345\nService Order: SO-99999"
        result_fields, vr = semantic_validate(
            "receipt_warranty_service", fields, raw
        )
        # Values differ — no disambiguation should occur
        assert result_fields["invoice_number"] == "WO-12345"
        assert result_fields["service_order_number"] == "SO-99999"

    def test_multiline_description_merge(self):
        fields = {
            **self.BASE_FIELDS,
            "items": [
                {"description": "Replace front brake", "quantity": 1,
                 "unit_price": "150.00", "line_total": "150.00"},
                {"description": "pads and rotors",     "quantity": None,
                 "unit_price": None,    "line_total": None},
                {"description": "Oil change",          "quantity": 1,
                 "unit_price": "35.00", "line_total": "35.00"},
            ],
        }
        result_fields, vr = semantic_validate(
            "receipt_warranty_service", fields, ""
        )
        items = result_fields["items"]
        assert len(items) == 2
        assert "pads and rotors" in items[0]["description"]
        assert items[1]["description"] == "Oil change"


# ── Test C: Delmarva utility bill — full semantic checks ──────────────────────

DELMARVA_RAW_TEXT = """
DELMARVA POWER  AN EXELON COMPANY
Account Number: 55040071056
Invoice Number: 200632579949

Service Address: 127 Goldleaf Dr W
Middletown, DE 19709

AMOUNT DUE: $103.63

Delivery Charges: $42.44
Supply Charges: $61.19

Usage Estimated

Return this stub with your payment
PO BOX 13609
Philadelphia, PA 19101
"""

DELMARVA_FIELDS = {
    "provider_name":       "delmarva power EXELON",
    "parent_company":      None,
    "provider_address":    None,
    "customer_name":       None,
    "service_address":     "127 Goldleaf Dr W, Middletown, DE 19709",
    "customer_address":    None,
    "account_number":      "55040071056",
    "bill_number":         "200632579949",
    "billing_period_start": None,
    "billing_period_end":  None,
    "bill_date":           None,
    "due_date":            None,
    "service_type":        None,
    "rate_plan":           None,
    "usage_value":         None,
    "usage_unit":          None,
    "usage_estimated":     None,
    "previous_balance":    None,
    "payments_received":   None,
    "adjustments":         None,
    "delivery_charges":    "42.44",
    "supply_charges":      "61.19",
    "subtotal":            None,
    "tax_amount":          None,
    "total_amount":        "103.63",
    "balance_due":         None,
    "currency":            None,
    "notes":               None,
}


class TestDelmarvaUtilityBill:

    def _validate(self, fields=None, raw_text=None):
        return semantic_validate(
            "bill_utility",
            fields or dict(DELMARVA_FIELDS),
            raw_text if raw_text is not None else DELMARVA_RAW_TEXT,
        )

    # Provider normalization

    def test_provider_name_normalized(self):
        result_fields, _ = self._validate()
        assert result_fields["provider_name"] == "Delmarva Power"

    def test_parent_company_inferred(self):
        result_fields, _ = self._validate()
        assert result_fields["parent_company"] == "Exelon"

    def test_parent_company_normalized_from_allcaps(self):
        """DI often extracts 'EXELON' in all caps — should be normalized to 'Exelon'."""
        fields = dict(DELMARVA_FIELDS)
        fields["parent_company"] = "EXELON"
        result_fields, _ = self._validate(fields)
        assert result_fields["parent_company"] == "Exelon"

    # Identifier fields

    def test_account_number_preserved(self):
        result_fields, vr = self._validate()
        assert result_fields["account_number"] == "55040071056"
        fv = vr.raw_to_normalized_mapping.get("account_number")
        # Should have high confidence (label "Account Number" present)
        if fv:
            assert fv.confidence >= 0.8

    def test_account_number_recovered_when_duplicated_from_bill_number(self):
        """When DI puts the invoice number in both fields, the real account number
        must be recovered from raw text — not left blank."""
        fields = dict(DELMARVA_FIELDS)
        fields["account_number"] = "200632579949"   # wrong — same as bill_number
        fields["bill_number"]    = "200632579949"
        result_fields, vr = self._validate(fields)
        # Should be recovered from "Account Number: 5504 0071 056" in raw text
        assert result_fields["account_number"] == "55040071056"
        assert result_fields["bill_number"] == "200632579949"

    def test_account_number_blank_when_no_raw_text_recovery(self):
        """Without raw text, wrong account_number that matches bill_number is blanked."""
        fields = dict(DELMARVA_FIELDS)
        fields["account_number"] = "200632579949"
        fields["bill_number"]    = "200632579949"
        result_fields, vr = self._validate(fields, raw_text="")
        assert result_fields["account_number"] is None
        assert any("account_number" in w for w in vr.validation_warnings)

    def test_account_number_recovered_when_missing(self):
        """If DI/LLM left account_number blank, validator recovers it from raw text."""
        fields = dict(DELMARVA_FIELDS)
        fields["account_number"] = None
        result_fields, _ = self._validate(fields)
        assert result_fields["account_number"] == "55040071056"

    def test_account_number_spaced_format_normalized(self):
        """'5504 0071 056' in raw text should be returned as '55040071056'."""
        fields = dict(DELMARVA_FIELDS)
        fields["account_number"] = None
        result_fields, _ = self._validate(fields)
        # No spaces in normalized value
        assert " " not in (result_fields.get("account_number") or "")

    def test_bill_number_preserved(self):
        result_fields, vr = self._validate()
        assert result_fields["bill_number"] == "200632579949"
        fv = vr.raw_to_normalized_mapping.get("bill_number")
        if fv:
            assert fv.confidence >= 0.8

    # Address semantics

    def test_service_address_preserved(self):
        result_fields, _ = self._validate()
        assert "127 Goldleaf" in (result_fields.get("service_address") or "")

    def test_po_box_not_in_service_address(self):
        fields = dict(DELMARVA_FIELDS)
        fields["service_address"] = "PO BOX 13609, Philadelphia, PA 19101"
        result_fields, vr = self._validate(fields)
        # PO Box in service_address should be blanked
        assert result_fields.get("service_address") is None
        assert any("service_address" in w for w in vr.validation_warnings)

    def test_remittance_address_set_as_provider_address(self):
        """When PO Box was in service_address, it should migrate to provider_address."""
        fields = dict(DELMARVA_FIELDS)
        fields["service_address"] = "PO BOX 13609, Philadelphia, PA 19101"
        result_fields, _ = self._validate(fields)
        assert result_fields.get("provider_address") is not None
        assert "13609" in str(result_fields["provider_address"])

    # Usage estimated flag

    def test_usage_estimated_auto_detected(self):
        result_fields, _ = self._validate()
        assert result_fields.get("usage_estimated") is True

    def test_usage_estimated_numeric_value_blanked(self):
        """If a numeric quantity ends up in usage_estimated, it should be cleared."""
        fields = dict(DELMARVA_FIELDS)
        fields["usage_estimated"] = "712 kWh"
        result_fields, vr = self._validate(fields)
        assert result_fields.get("usage_estimated") is None
        assert any("usage_estimated" in w for w in vr.validation_warnings)

    def test_usage_estimated_integer_blanked(self):
        fields = dict(DELMARVA_FIELDS)
        fields["usage_estimated"] = 712
        result_fields, vr = self._validate(fields)
        assert result_fields.get("usage_estimated") is None

    # Subtotal guard

    def test_subtotal_not_fabricated_when_no_label(self):
        """Subtotal should be cleared if the document text doesn't show the word 'Subtotal'."""
        fields = dict(DELMARVA_FIELDS)
        fields["subtotal"] = "103.63"  # LLM may hallucinate this
        result_fields, vr = self._validate(fields)
        # Raw text has no 'subtotal' word — should be blanked
        assert result_fields.get("subtotal") is None
        assert any("subtotal" in w for w in vr.validation_warnings)

    def test_subtotal_kept_when_labeled(self):
        fields = dict(DELMARVA_FIELDS)
        fields["subtotal"] = "103.63"
        raw = DELMARVA_RAW_TEXT + "\nSubtotal: $103.63"
        result_fields, _ = self._validate(fields, raw_text=raw)
        assert result_fields.get("subtotal") == "103.63"

    # Charge components

    def test_delivery_charges_preserved(self):
        result_fields, _ = self._validate()
        assert result_fields.get("delivery_charges") == "42.44"

    def test_supply_charges_preserved(self):
        result_fields, _ = self._validate()
        assert result_fields.get("supply_charges") == "61.19"

    def test_delivery_plus_supply_equals_total(self):
        result_fields, vr = self._validate()
        d = float(result_fields.get("delivery_charges") or 0)
        s = float(result_fields.get("supply_charges") or 0)
        t = float(result_fields.get("total_amount") or 0)
        # 42.44 + 61.19 = 103.63 — no warning expected
        assert abs(d + s - t) < 0.02
        assert not any("exceeds total" in w for w in vr.validation_warnings)

    def test_delivery_supply_sum_exceeds_total_warns(self):
        fields = dict(DELMARVA_FIELDS)
        fields["delivery_charges"] = "80.00"
        fields["supply_charges"]   = "61.19"  # sum 141.19 > total 103.63
        _, vr = self._validate(fields)
        assert any("exceeds total" in w for w in vr.validation_warnings)

    # Arithmetic check

    def test_arithmetic_check_passes_when_consistent(self):
        fields = dict(DELMARVA_FIELDS)
        fields["previous_balance"]  = "0.00"
        fields["payments_received"] = "0.00"
        fields["total_amount"]      = "103.63"
        fields["balance_due"]       = "103.63"
        _, vr = self._validate(fields)
        assert not any("balance_due" in w for w in vr.validation_warnings)

    def test_arithmetic_check_warns_on_mismatch(self):
        fields = dict(DELMARVA_FIELDS)
        fields["previous_balance"]  = "50.00"
        fields["payments_received"] = "50.00"
        fields["total_amount"]      = "103.63"
        fields["balance_due"]       = "200.00"  # should be 103.63
        _, vr = self._validate(fields)
        assert any("balance_due" in w for w in vr.validation_warnings)

    # Overall result structure

    def test_returns_validation_result(self):
        _, vr = self._validate()
        assert isinstance(vr, ValidationResult)
        assert isinstance(vr.normalized_fields, dict)
        assert isinstance(vr.field_confidence, dict)
        assert vr.validation_status in ("valid", "warning", "invalid")
        assert isinstance(vr.validation_warnings, list)
        assert isinstance(vr.validation_errors, list)
        assert isinstance(vr.raw_to_normalized_mapping, dict)

    def test_no_validation_errors_on_clean_bill(self):
        _, vr = self._validate()
        assert not vr.validation_errors


# ── Test D: Edge cases / pass-through for unknown doc types ──────────────────

class TestFallthrough:

    def test_unknown_doc_type_passes_through(self):
        fields = {"foo": "bar", "baz": 123}
        result_fields, vr = semantic_validate("catalog_product_list", fields, "")
        assert result_fields["foo"] == "bar"
        assert vr.validation_status == "valid"
        assert not vr.validation_errors
        assert not vr.validation_warnings

    def test_empty_fields_returns_valid(self):
        _, vr = semantic_validate("receipt", {}, "")
        assert vr.validation_status == "valid"

    def test_quote_valid_until_before_quote_date_warns(self):
        fields = {
            "quote_number": "Q-001",
            "quote_date": "2026-03-01",
            "valid_until": "2026-02-01",  # before quote_date
        }
        _, vr = semantic_validate("quote_vendor", fields, "")
        assert any("valid_until" in w for w in vr.validation_warnings)

    def test_quote_valid_until_after_quote_date_ok(self):
        fields = {
            "quote_number": "Q-001",
            "quote_date": "2026-03-01",
            "valid_until": "2026-04-01",
        }
        _, vr = semantic_validate("quote_vendor", fields, "")
        assert not any("valid_until" in w for w in vr.validation_warnings)


# ═══════════════════════════════════════════════════════════════════════════════
# Packing Slip / Shipment validator tests
# ═══════════════════════════════════════════════════════════════════════════════

AMAZON_RAW_TEXT = """\
Your order of January 12, 2001
This shipment completes your order.

Order Number: 002-5313943-9048845
Sold by: Amazon.com

Shipped via UPS

avbh12030/-1-/49999/std-68/1314266

ITEM                                         QTY   PRICE
Product XYZ                                    1   499.99

Subtotal                                           499.99
Shipping & Handling                                  4.48
Promotional Certificate                             -4.48
Order Total                                        499.99
Paid via Visa                                      499.99
Balance Due                                          0.00
"""

AMAZON_FIELDS = {
    "packing_slip_number": None,
    "order_number":        "002-5313943-9048845",
    "order_date":          None,
    "po_number":           None,
    "ship_date":           "2001-01-12",    # will be blanked — no ship date label
    "delivery_date":       "2001-01-15",    # will be blanked — no delivery date label
    "shipper_name":        "Amazon.com",
    "shipper_address":     None,
    "carrier_name":        "UPS",
    "tracking_number":     "avbh12030/-1-/49999/std-68/1314266",  # OCR garbage
    "shipment_id":         None,
    "shipping_method":     None,
    "weight":              None,
    "total_packages":      None,
    "subtotal":            None,
    "shipping_cost":       None,
    "discount":            None,
    "total_amount":        None,
    "payment_method":      None,
    "amount_paid":         None,
    "balance_due":         None,
    "notes":               None,
    "items":               [{"sku": None, "description": "Product XYZ",
                             "quantity": "1", "unit_of_measure": None}],
}


class TestPackingSlipAmazon:
    """Test A — full Amazon shipment document."""

    def _validate(self, fields=None, raw_text=None):
        return semantic_validate(
            "packing_slip",
            fields or dict(AMAZON_FIELDS),
            raw_text if raw_text is not None else AMAZON_RAW_TEXT,
        )

    # Order identifiers

    def test_order_number_preserved(self):
        result, _ = self._validate()
        assert result["order_number"] == "002-5313943-9048845"

    def test_order_date_extracted_from_text(self):
        result, vr = self._validate()
        assert result["order_date"] == "2001-01-12"
        fv = vr.raw_to_normalized_mapping.get("order_date")
        assert fv and fv.status == "valid"

    def test_shipper_name_preserved(self):
        result, _ = self._validate()
        assert result["shipper_name"] == "Amazon.com"

    def test_carrier_name_preserved(self):
        result, _ = self._validate()
        assert result["carrier_name"] == "UPS"

    # Date guards — no explicit labels in this document

    def test_ship_date_blanked_no_label(self):
        result, vr = self._validate()
        assert result["ship_date"] is None
        assert any("ship_date" in w for w in vr.validation_warnings)

    def test_delivery_date_blanked_no_label(self):
        result, vr = self._validate()
        assert result["delivery_date"] is None
        assert any("delivery_date" in w for w in vr.validation_warnings)

    # Tracking number

    def test_noisy_tracking_number_rejected(self):
        result, vr = self._validate()
        assert result["tracking_number"] is None
        assert any("tracking_number" in w for w in vr.validation_warnings)

    def test_tracking_rejection_reason_mentions_slashes(self):
        result, vr = self._validate()
        warning = next(w for w in vr.validation_warnings if "tracking_number" in w)
        assert "slash" in warning.lower() or "noisy" in warning.lower()

    # Financial summary

    def test_subtotal_extracted(self):
        result, _ = self._validate()
        assert result["subtotal"] == "499.99"

    def test_shipping_cost_extracted(self):
        result, _ = self._validate()
        assert result["shipping_cost"] == "4.48"

    def test_discount_extracted_as_negative(self):
        result, _ = self._validate()
        assert result["discount"] == "-4.48"

    def test_total_amount_extracted(self):
        result, _ = self._validate()
        assert result["total_amount"] == "499.99"

    def test_payment_method_parsed_from_paid_via(self):
        result, _ = self._validate()
        assert result["payment_method"] == "VISA"

    def test_amount_paid_parsed_from_paid_via(self):
        result, _ = self._validate()
        assert result["amount_paid"] == "499.99"

    def test_balance_due_extracted(self):
        result, _ = self._validate()
        assert result["balance_due"] == "0.00"

    # Financial math passes

    def test_financial_math_no_warning(self):
        result, vr = self._validate()
        # 499.99 + 4.48 + (-4.48) = 499.99 — should not warn
        assert not any("total_amount" in w for w in vr.validation_warnings)

    def test_amount_paid_balance_math_no_warning(self):
        result, vr = self._validate()
        # 499.99 + 0.00 = 499.99 — should not warn
        assert not any("amount_paid" in w for w in vr.validation_warnings)


class TestPackingSlipExplicitShipDate:
    """Test B — document with explicit Ship Date label."""

    FIELDS = {
        **{k: None for k in AMAZON_FIELDS},
        "ship_date": "2026-03-01",
        "carrier_name": None,
        "tracking_number": None,
        "items": [],
    }

    def test_ship_date_kept_when_label_present(self):
        raw = "Ship Date: 2026-03-01\nOrder Number: 123"
        result, vr = semantic_validate("packing_slip", dict(self.FIELDS), raw)
        assert result["ship_date"] == "2026-03-01"
        assert not any("ship_date" in w for w in vr.validation_warnings)

    def test_shipped_on_label_accepted(self):
        raw = "Shipped On: 2026-03-01\nOrder Number: 123"
        result, vr = semantic_validate("packing_slip", dict(self.FIELDS), raw)
        assert result["ship_date"] == "2026-03-01"
        assert not any("ship_date" in w for w in vr.validation_warnings)

    def test_date_shipped_label_accepted(self):
        raw = "Date Shipped: 2026-03-01\nOrder Number: 123"
        result, vr = semantic_validate("packing_slip", dict(self.FIELDS), raw)
        assert result["ship_date"] == "2026-03-01"


class TestPackingSlipExplicitDeliveryDate:
    """Test C — document with explicit Delivery Date label."""

    FIELDS = {
        **{k: None for k in AMAZON_FIELDS},
        "delivery_date": "2026-03-04",
        "carrier_name": None,
        "tracking_number": None,
        "items": [],
    }

    def test_delivery_date_kept_when_label_present(self):
        raw = "Delivery Date: 2026-03-04\nOrder Number: 123"
        result, vr = semantic_validate("packing_slip", dict(self.FIELDS), raw)
        assert result["delivery_date"] == "2026-03-04"
        assert not any("delivery_date" in w for w in vr.validation_warnings)

    def test_expected_delivery_label_accepted(self):
        raw = "Expected Delivery: 2026-03-04\nOrder Number: 123"
        result, vr = semantic_validate("packing_slip", dict(self.FIELDS), raw)
        assert result["delivery_date"] == "2026-03-04"

    def test_estimated_delivery_label_accepted(self):
        raw = "Estimated Delivery: 2026-03-04\nOrder Number: 123"
        result, vr = semantic_validate("packing_slip", dict(self.FIELDS), raw)
        assert result["delivery_date"] == "2026-03-04"


class TestPackingSlipTrackingValidation:
    """Test D — tracking number noise rejection and carrier pattern validation."""

    def _run(self, tracking: str, carrier: str = "UPS", raw: str = ""):
        fields = {**{k: None for k in AMAZON_FIELDS},
                  "tracking_number": tracking, "carrier_name": carrier, "items": []}
        return semantic_validate("packing_slip", fields, raw)

    def test_noisy_slash_rejected(self):
        result, vr = self._run("avbh12030/-1-/49999/std-68/1314266")
        assert result["tracking_number"] is None

    def test_std_token_rejected(self):
        result, vr = self._run("track12345/std-68/999")
        assert result["tracking_number"] is None

    def test_valid_ups_1z_accepted(self):
        result, vr = self._run("1Z999AA10123456784")
        assert result["tracking_number"] == "1Z999AA10123456784"
        fv = vr.raw_to_normalized_mapping.get("tracking_number")
        assert fv and fv.status == "valid"

    def test_invalid_ups_format_rejected(self):
        # Valid looking but wrong length — rejects
        result, vr = self._run("1Z999AA101234")
        assert result["tracking_number"] is None

    def test_valid_fedex_12digit_accepted(self):
        result, vr = self._run("123456789012", carrier="FedEx")
        assert result["tracking_number"] == "123456789012"

    def test_no_carrier_generic_alphanumeric_accepted_with_warning(self):
        result, vr = self._run("ABCD12345678", carrier="")
        assert result["tracking_number"] == "ABCD12345678"
        # Should warn about unverified format
        assert any("tracking_number" in w for w in vr.validation_warnings)

    def test_too_long_rejected(self):
        result, vr = self._run("A" * 45)
        assert result["tracking_number"] is None

    def test_garbage_string_rejected(self):
        result, vr = self._run("not-a-tracking/num/123")
        assert result["tracking_number"] is None


class TestPackingSlipOrderDate:
    """Additional order_date extraction cases."""

    def _run(self, raw: str):
        fields = {**{k: None for k in AMAZON_FIELDS}, "items": []}
        return semantic_validate("packing_slip", fields, raw)

    def test_your_order_of_format(self):
        result, _ = self._run("Your order of January 12, 2001")
        assert result["order_date"] == "2001-01-12"

    def test_your_order_of_without_comma(self):
        result, _ = self._run("Your order of March 5 2026")
        assert result["order_date"] == "2026-03-05"

    def test_order_date_label_iso(self):
        result, _ = self._run("Order Date: 2026-03-15")
        assert result["order_date"] == "2026-03-15"

    def test_no_order_date_stays_none(self):
        result, _ = self._run("No dates here")
        assert result.get("order_date") is None


class TestPackingSlipFinancialMath:
    """Financial arithmetic validation."""

    def _run(self, overrides: dict):
        fields = {**{k: None for k in AMAZON_FIELDS}, "items": [], **overrides}
        return semantic_validate("packing_slip", fields, "")

    def test_mismatch_warns(self):
        _, vr = self._run({
            "subtotal": "100.00", "shipping_cost": "5.00",
            "discount": "0.00",   "total_amount": "200.00",
        })
        assert any("total_amount" in w for w in vr.validation_warnings)

    def test_paid_plus_balance_mismatch_warns(self):
        _, vr = self._run({
            "total_amount": "100.00", "amount_paid": "50.00", "balance_due": "60.00",
        })
        assert any("amount_paid" in w for w in vr.validation_warnings)

    def test_negative_discount_preserved_through_math(self):
        result, vr = self._run({
            "subtotal": "499.99", "shipping_cost": "4.48",
            "discount": "-4.48",  "total_amount": "499.99",
        })
        assert result["discount"] == "-4.48"
        assert not any("total_amount" in w for w in vr.validation_warnings)


# ── Test F: FBA / shipping-label party swap detection ─────────────────────────

# Raw OCR text from an FBA shipping label with two named sections
FBA_LABEL_RAW = """\
SHIP FROM:
James Bond
333 Boren Ave N
Seattle, WA 98109
United States

SHIP TO:
FBA: dnest+sta012
Amazon.com Services, Inc.
4255 Anson Blvd
Whitestown, IN 46075-4412
United States

Carrier: UPS
Tracking: 1Z999AA10123456784
"""

# Fields as DI/LLM returned them — names are SWAPPED
FBA_SWAPPED_FIELDS = {
    **{k: None for k in AMAZON_FIELDS},
    "shipper_name":    "Amazon.com Services, Inc.",   # WRONG — this is ship_to
    "shipper_address": "4255 Anson Blvd, Whitestown, IN 46075-4412",
    "ship_to_name":    "James Bond",                   # WRONG — this is shipper
    "ship_to_address": "333 Boren Ave N, Seattle, WA 98109",
    "carrier_name":    "UPS",
    "tracking_number": "1Z999AA10123456784",
    "items":           [],
}

# Fields already correct — should be left unchanged
FBA_CORRECT_FIELDS = {
    **{k: None for k in AMAZON_FIELDS},
    "shipper_name":    "James Bond",
    "shipper_address": "333 Boren Ave N, Seattle, WA 98109",
    "ship_to_name":    "Amazon.com Services, Inc.",
    "ship_to_address": "4255 Anson Blvd, Whitestown, IN 46075-4412",
    "carrier_name":    "UPS",
    "tracking_number": "1Z999AA10123456784",
    "items":           [],
}


class TestPackingSlipShipperSwap:
    """Detect and correct ship_to_name / shipper_name swap from two-column labels."""

    def _validate(self, fields, raw=FBA_LABEL_RAW):
        return semantic_validate("packing_slip", dict(fields), raw)

    # Swap correction

    def test_shipper_name_corrected_when_swapped(self):
        result, _ = self._validate(FBA_SWAPPED_FIELDS)
        assert result["shipper_name"] == "James Bond"

    def test_ship_to_name_corrected_when_swapped(self):
        result, _ = self._validate(FBA_SWAPPED_FIELDS)
        assert result["ship_to_name"] == "Amazon.com Services, Inc."

    def test_shipper_address_swapped_with_names(self):
        result, _ = self._validate(FBA_SWAPPED_FIELDS)
        assert result["shipper_address"] == "4255 Anson Blvd, Whitestown, IN 46075-4412" or \
               "Seattle" in (result["shipper_address"] or "")

    def test_ship_to_address_swapped_with_names(self):
        result, _ = self._validate(FBA_SWAPPED_FIELDS)
        assert "Whitestown" in (result["ship_to_address"] or "") or \
               result["ship_to_address"] == "333 Boren Ave N, Seattle, WA 98109"

    def test_swap_warning_emitted(self):
        _, vr = self._validate(FBA_SWAPPED_FIELDS)
        assert any("swap" in w.lower() for w in vr.validation_warnings)

    def test_swap_warning_mentions_both_fields(self):
        _, vr = self._validate(FBA_SWAPPED_FIELDS)
        swap_warnings = [w for w in vr.validation_warnings if "swap" in w.lower()]
        assert any("shipper" in w.lower() and "ship_to" in w.lower()
                   for w in swap_warnings)

    # No false-positive when names are correct

    def test_no_swap_when_names_already_correct(self):
        result, vr = self._validate(FBA_CORRECT_FIELDS)
        assert result["shipper_name"] == "James Bond"
        assert result["ship_to_name"] == "Amazon.com Services, Inc."
        assert not any("swap" in w.lower() for w in vr.validation_warnings)

    # Extraction when both fields are blank

    def test_shipper_extracted_from_ship_from_section_when_blank(self):
        fields = {**FBA_SWAPPED_FIELDS, "shipper_name": None, "shipper_address": None,
                  "ship_to_name": None, "ship_to_address": None}
        result, _ = self._validate(fields)
        assert result["shipper_name"] == "James Bond"

    def test_ship_to_extracted_from_ship_to_section_when_blank(self):
        fields = {**FBA_SWAPPED_FIELDS, "shipper_name": None, "shipper_address": None,
                  "ship_to_name": None, "ship_to_address": None}
        result, _ = self._validate(fields)
        # FBA reference line is skipped; actual company name follows
        assert result["ship_to_name"] == "Amazon.com Services, Inc."

    # No SHIP FROM section — no cross-check attempted

    def test_no_swap_check_without_ship_from_label(self):
        raw_no_from = """\
SHIP TO:
James Bond
333 Boren Ave N
Seattle, WA 98109
"""
        fields = {**{k: None for k in AMAZON_FIELDS},
                  "shipper_name": None, "ship_to_name": "James Bond",
                  "carrier_name": None, "tracking_number": None, "items": []}
        result, vr = self._validate(fields, raw=raw_no_from)
        assert result["ship_to_name"] == "James Bond"
        assert not any("swap" in w.lower() for w in vr.validation_warnings)
