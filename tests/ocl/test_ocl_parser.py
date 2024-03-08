import sys
from antlr4 import *
from besser.BUML.notations.ocl.OCLsLexer import OCLsLexer
from besser.BUML.notations.ocl.OCLsParser import OCLsParser
from besser.BUML.notations.ocl.OCLsListener import OCLsListener
import unittest

class TestOclParser(unittest.TestCase):
# from OCLInterp import OCLInterp

    def test_derive(self):
        ocl = "context LoyaltyAccount::totalPointsEarned : Integer derive :	self.transactions->select( i_Transaction : Transaction | i_Transaction.oclIsTypeOf(Earning) )->collect( i_Transaction : Transaction | i_Transaction.points )->sum()  ;"
        input_stream = InputStream(ocl)
        lexer = OCLsLexer(input_stream)
        stream = CommonTokenStream(lexer)
        parser = OCLsParser(stream)
        tree = parser.oclFile()
        assert parser.getNumberOfSyntaxErrors() == 0
    def test_seq(self):
        ocl ="context ServiceLevel inv invariant_ServiceLevel19 :	(Sequence{'a', 'b', 'c', 'c', 'd', 'e'}->prepend('X')) = Sequence{'X', 'a', 'b', 'c', 'c', 'd', 'e'}"
        #ocl ="context ServiceLevel inv invariant_ServiceLevel19 :	(Sequence{'a', 'b', 'c', 'c', 'd', 'e'}->prepend('X'))"

        input_stream = InputStream(ocl)
        lexer = OCLsLexer(input_stream)
        stream = CommonTokenStream(lexer)
        parser = OCLsParser(stream)
        tree = parser.oclFile()
        assert parser.getNumberOfSyntaxErrors() == 0
    def test_ordered_set(self):
        # ocl = "context S inv invariant_ServiceLevel17 :(OrderedSet{'a', 'b', 'c', 'd'}->subOrderedSet(2, 3)) = OrderedSet{'b', 'c'}"
        ocl = "context S inv invariant_ServiceLevel17 :(OrderedSet{'a', 'b', 'c', 'd'}->subOrderedSet(2, 3)) = OrderedSet{'b', 'c'}"
        input_stream = InputStream(ocl)
        lexer = OCLsLexer(input_stream)
        stream = CommonTokenStream(lexer)
        parser = OCLsParser(stream)
        tree = parser.oclFile()
        assert parser.getNumberOfSyntaxErrors() == 0
    def test_is_empty(self):
        ocl = "context s inv invariant_ServiceLevel4 :	Bag{Set{1, 2}, Set{1, 2}, Set{4, 5, 6}}->isEmpty()"
        input_stream = InputStream(ocl)
        lexer = OCLsLexer(input_stream)
        stream = CommonTokenStream(lexer)
        parser = OCLsParser(stream)
        tree = parser.oclFile()
        assert parser.getNumberOfSyntaxErrors() == 0
    def test_last(self):
        ocl = "context s inv invariant_ServiceLevel12 :	(OrderedSet{'a', 'b', 'c', 'd'}->last()) = 'd'"
        input_stream = InputStream(ocl)
        lexer = OCLsLexer(input_stream)
        stream = CommonTokenStream(lexer)
        parser = OCLsParser(stream)
        tree = parser.oclFile()
        assert parser.getNumberOfSyntaxErrors() == 0
    def test_append(self):
        ocl = "context s inv invariant_ServiceLevel18 :(Sequence{'a', 'b', 'c', 'c', 'd', 'e'}->append('X')) = Sequence{'a', 'b', 'c', 'c', 'd', 'e', 'X'}"
        input_stream = InputStream(ocl)
        lexer = OCLsLexer(input_stream)
        stream = CommonTokenStream(lexer)
        parser = OCLsParser(stream)
        tree = parser.oclFile()
        assert parser.getNumberOfSyntaxErrors() == 0
    def test_is_empty_2(self):
        ocl = "context s inv invariant_ServiceLevel1 :	self.program.partners->isEmpty()"
        input_stream = InputStream(ocl)
        lexer = OCLsLexer(input_stream)
        stream = CommonTokenStream(lexer)
        parser = OCLsParser(stream)
        tree = parser.oclFile()
        assert parser.getNumberOfSyntaxErrors() == 0
    def test_symDiff(self):
        ocl = "context s inv invariant_ServiceLevel10 :	(Set{1, 4, 7, 10}->symmetricDifference(Set{4, 5, 7})) = Set{1, 5, 10}"
        input_stream = InputStream(ocl)
        lexer = OCLsLexer(input_stream)
        stream = CommonTokenStream(lexer)
        parser = OCLsParser(stream)
        tree = parser.oclFile()
        assert parser.getNumberOfSyntaxErrors() == 0
    def test_is_empty_3(self):
        ocl = "context s inv invariant_ServiceLevel7 :	Sequence{2, 1, 2, 3, 5, 6, 4}->isEmpty()"
        input_stream = InputStream(ocl)
        lexer = OCLsLexer(input_stream)
        stream = CommonTokenStream(lexer)
        parser = OCLsParser(stream)
        tree = parser.oclFile()
        assert parser.getNumberOfSyntaxErrors() == 0
    def test_is_empty_4(self):
        ocl = "context s inv invariant_ServiceLevel5 :	Bag{1, 1, 2, 2, 4, 5, 6}->isEmpty()"
        input_stream = InputStream(ocl)
        lexer = OCLsLexer(input_stream)
        stream = CommonTokenStream(lexer)
        parser = OCLsParser(stream)
        tree = parser.oclFile()
        assert parser.getNumberOfSyntaxErrors() == 0
    def test_subseq (self):
        ocl = "context s inv invariant_ServiceLevel16 :	(Sequence{'a', 'b', 'c', 'c', 'd', 'e'}->subSequence(3, 5)) = Sequence{'c', 'c', 'd'}"
        input_stream = InputStream(ocl)
        lexer = OCLsLexer(input_stream)
        stream = CommonTokenStream(lexer)
        parser = OCLsParser(stream)
        tree = parser.oclFile()
        assert parser.getNumberOfSyntaxErrors() == 0
    def test_is_empty_5(self):
        ocl = "context s inv invariant_ServiceLevel6 :	Sequence{Set{1, 2}, Set{2, 3}, Set{4, 5, 6}}->isEmpty()"
        input_stream = InputStream(ocl)
        lexer = OCLsLexer(input_stream)
        stream = CommonTokenStream(lexer)
        parser = OCLsParser(stream)
        tree = parser.oclFile()
        assert parser.getNumberOfSyntaxErrors() == 0
    def test_at(self):
        ocl = "context s inv invariant_ServiceLevel13 :	(Sequence{'a', 'b', 'c', 'c', 'd', 'e'}->at(3)) = 'c'"
        input_stream = InputStream(ocl)
        lexer = OCLsLexer(input_stream)
        stream = CommonTokenStream(lexer)
        parser = OCLsParser(stream)
        tree = parser.oclFile()
        assert parser.getNumberOfSyntaxErrors() == 0
    def test_is_empty_5(self):
        ocl = "context s inv invariant_ServiceLevel3 :	Set{1, 2, 3, 4, 5, 6}->isEmpty()"
        input_stream = InputStream(ocl)
        lexer = OCLsLexer(input_stream)
        stream = CommonTokenStream(lexer)
        parser = OCLsParser(stream)
        tree = parser.oclFile()
        assert parser.getNumberOfSyntaxErrors() == 0
    def test_first(self):
        ocl = "context s inv invariant_ServiceLevel11 :	(Sequence{'a', 'b', 'c', 'c', 'd', 'e'}->first()) = 'a'"
        input_stream = InputStream(ocl)
        lexer = OCLsLexer(input_stream)
        stream = CommonTokenStream(lexer)
        parser = OCLsParser(stream)
        tree = parser.oclFile()
        assert parser.getNumberOfSyntaxErrors() == 0
    def test_minus(self):
        ocl = "context s inv invariant_ServiceLevel8 :	((Set{1, 4, 7, 10}) - Set{4, 7}) = Set{1, 10}"
        input_stream = InputStream(ocl)
        lexer = OCLsLexer(input_stream)
        stream = CommonTokenStream(lexer)
        parser = OCLsParser(stream)
        tree = parser.oclFile()
        assert parser.getNumberOfSyntaxErrors() == 0
    def test_is_empty_6(self):
        ocl = "context s inv invariant_ServiceLevel2 :	Set{Set{1, 2}, Set{2, 3}, Set{4, 5, 6}}->isEmpty() "
        input_stream = InputStream(ocl)
        lexer = OCLsLexer(input_stream)
        stream = CommonTokenStream(lexer)
        parser = OCLsParser(stream)
        tree = parser.oclFile()
        assert parser.getNumberOfSyntaxErrors() == 0
    def test_ordered_set_2(self):
        ocl = "context s inv invariant_ServiceLevel9 :	((OrderedSet{12, 9, 6, 3}) - Set{1, 3, 2}) = OrderedSet{12, 9, 6}"
        input_stream = InputStream(ocl)
        lexer = OCLsLexer(input_stream)
        stream = CommonTokenStream(lexer)
        parser = OCLsParser(stream)
        tree = parser.oclFile()
        assert parser.getNumberOfSyntaxErrors() == 0
    def test_index_of(self):
        ocl = "context s inv invariant_ServiceLevel14 :	(Sequence{'a', 'b', 'c', 'c', 'd', 'e'}->indexOf('c')) = 3"
        input_stream = InputStream(ocl)
        lexer = OCLsLexer(input_stream)
        stream = CommonTokenStream(lexer)
        parser = OCLsParser(stream)
        tree = parser.oclFile()
        assert parser.getNumberOfSyntaxErrors() == 0
    def test_insert_at(self):
        ocl = "context s inv invariant_ServiceLevel15 :	(OrderedSet{'a', 'b', 'c', 'd'}->insertAt(3, 'X')) = OrderedSet{'a', 'b', 'X', 'c', 'd'} "
        input_stream = InputStream(ocl)
        lexer = OCLsLexer(input_stream)
        stream = CommonTokenStream(lexer)
        parser = OCLsParser(stream)
        tree = parser.oclFile()
        assert parser.getNumberOfSyntaxErrors() == 0
    def test_is_kind_of(self):
        ocl = "context Transaction inv ianvariant_Transaction1 :	self.oclIsKindOf(Transaction) = true"
        input_stream = InputStream(ocl)
        lexer = OCLsLexer(input_stream)
        stream = CommonTokenStream(lexer)
        parser = OCLsParser(stream)
        tree = parser.oclFile()
        assert parser.getNumberOfSyntaxErrors() == 0
    def test_is_type_of(self):
        ocl = "context T inv invariant_Transaction3 :self.oclIsTypeOf(Burning) = false"
        input_stream = InputStream(ocl)
        lexer = OCLsLexer(input_stream)
        stream = CommonTokenStream(lexer)
        parser = OCLsParser(stream)
        tree = parser.oclFile()
        assert parser.getNumberOfSyntaxErrors() == 0
    def test_is_type_of_2(self):
        ocl= "context T inv invariant_Transaction2 :	self.oclIsTypeOf(Transaction) = true"
        input_stream = InputStream(ocl)
        lexer = OCLsLexer(input_stream)
        stream = CommonTokenStream(lexer)
        parser = OCLsParser(stream)
        tree = parser.oclFile()
        assert parser.getNumberOfSyntaxErrors() == 0
    def test_is_kind_of_1(self):
        ocl = "context T inv invariant_Transaction4 : self.oclIsKindOf(Burning) = false"
        input_stream = InputStream(ocl)
        lexer = OCLsLexer(input_stream)
        stream = CommonTokenStream(lexer)
        parser = OCLsParser(stream)
        tree = parser.oclFile()
        assert parser.getNumberOfSyntaxErrors() == 0
    def test_post(self):
        ocl = "context Transaction::program() : LoyaltyProgram post:	result = self.card.Membership.programs"
        input_stream = InputStream(ocl)
        lexer = OCLsLexer(input_stream)
        stream = CommonTokenStream(lexer)
        parser = OCLsParser(stream)
        tree = parser.oclFile()
        assert parser.getNumberOfSyntaxErrors() == 0
    def test_implies(self):
        ocl = "context LoyaltyAccount inv invariant_points : (self.points > 0) implies self.transactions->exists( t : Transaction | t.points > 0 )"
        input_stream = InputStream(ocl)
        lexer = OCLsLexer(input_stream)
        stream = CommonTokenStream(lexer)
        parser = OCLsParser(stream)
        tree = parser.oclFile()
        assert parser.getNumberOfSyntaxErrors() == 0
    def test_collect(self):
        ocl = "context s inv invariant_transactions :	self.transactions->collect( i_Transaction : Transaction | i_Transaction.points )->exists( p : Integer | p = 500 )"
        input_stream = InputStream(ocl)
        lexer = OCLsLexer(input_stream)
        stream = CommonTokenStream(lexer)
        parser = OCLsParser(stream)
        tree = parser.oclFile()
        assert parser.getNumberOfSyntaxErrors() == 0
    def test_collect_2(self):
        ocl = "context s inv invariant_oneOwner :	(self.transactions->collect( i_Transaction : Transaction | i_Transaction.card )->collect( i_CustomerCard : CustomerCard | i_CustomerCard.owner )->asSet()->size()) = 1"
        input_stream = InputStream(ocl)
        lexer = OCLsLexer(input_stream)
        stream = CommonTokenStream(lexer)
        parser = OCLsParser(stream)
        tree = parser.oclFile()
        assert parser.getNumberOfSyntaxErrors() == 0
    def test_init(self):
        ocl = "context s context LoyaltyAccount::points : Integer init :0"
        input_stream = InputStream(ocl)
        lexer = OCLsLexer(input_stream)
        stream = CommonTokenStream(lexer)
        parser = OCLsParser(stream)
        tree = parser.oclFile()
        assert parser.getNumberOfSyntaxErrors() == 0
    def test_derive_2(self):
        ocl = "context LoyaltyAccount::totalPointsEarned : Integer derive :	self.transactions->select( i_Transaction : Transaction | i_Transaction.oclIsTypeOf(Earning) )->collect( i_Transaction : Transaction | i_Transaction.points )->sum()"
        input_stream = InputStream(ocl)
        lexer = OCLsLexer(input_stream)
        stream = CommonTokenStream(lexer)
        parser = OCLsParser(stream)
        tree = parser.oclFile()
        assert parser.getNumberOfSyntaxErrors() == 0
    def test_derive_3(self):
        ocl = "context LoyaltyAccount::usedServices : Set(Service) derive :	self.transactions->collect( i_Transaction : Transaction | i_Transaction.generatedBy )->asSet()"
        input_stream = InputStream(ocl)
        lexer = OCLsLexer(input_stream)
        stream = CommonTokenStream(lexer)
        parser = OCLsParser(stream)
        tree = parser.oclFile()
        assert parser.getNumberOfSyntaxErrors() == 0
    def test_init_2(self):
        ocl = "context LoyaltyAccount::transactions : Set(Transaction) init : Set{}"
        input_stream = InputStream(ocl)
        lexer = OCLsLexer(input_stream)
        stream = CommonTokenStream(lexer)
        parser = OCLsParser(stream)
        tree = parser.oclFile()
        assert parser.getNumberOfSyntaxErrors() == 0
    def test_body(self):
        ocl = "context LoyaltyAccount::getCustomerName() : String body:	self.Membership.card.owner.name"
        input_stream = InputStream(ocl)
        lexer = OCLsLexer(input_stream)
        stream = CommonTokenStream(lexer)
        parser = OCLsParser(stream)
        tree = parser.oclFile()
        assert parser.getNumberOfSyntaxErrors() == 0
    def test_post_pre(self):
        ocl = "context LoyaltyAccount::isEmpty() : Boolean post testPostSuggestedName:	result = self.points = 0 pre testPreSuggestedName:	true"
        # ocl = "context LoyaltyAccount::isEmpty()"
        input_stream = InputStream(ocl)
        lexer = OCLsLexer(input_stream)
        stream = CommonTokenStream(lexer)
        parser = OCLsParser(stream)
        tree = parser.oclFile()
        assert parser.getNumberOfSyntaxErrors() == 0
    def test_collect(self):
        ocl = "context ProgramPartner inv invariant_totalPointsEarning2 :	(self.deliveredServices->collect( i_Service : Service | i_Service.transactions )->select( i_Transaction : Transaction | i_Transaction.oclIsTypeOf(Earning) )->collect( i_Transaction : Transaction | i_Transaction.points )->sum()) < 10000 "
        input_stream = InputStream(ocl)
        lexer = OCLsLexer(input_stream)
        stream = CommonTokenStream(lexer)
        parser = OCLsParser(stream)
        tree = parser.oclFile()
        assert parser.getNumberOfSyntaxErrors() == 0
    def test_collect_2(self):
        ocl = "context s inv invariant_totalPointsEarning :	(self.deliveredServices->collect( i_Service : Service | i_Service.transactions )->select( i_Transaction : Transaction | i_Transaction.oclIsTypeOf(Earning) )->collect( i_Transaction : Transaction | i_Transaction.points )->sum()) < 10000 "
        input_stream = InputStream(ocl)
        lexer = OCLsLexer(input_stream)
        stream = CommonTokenStream(lexer)
        parser = OCLsParser(stream)
        tree = parser.oclFile()
        assert parser.getNumberOfSyntaxErrors() == 0
    def test_collect_3(self):
        ocl = "context s inv invariant_nrOfParticipants :self.numberOfCustomers = self.programs->collect( i_LoyaltyProgram : LoyaltyProgram | i_LoyaltyProgram.participants )->size()"
        input_stream = InputStream(ocl)
        lexer = OCLsLexer(input_stream)
        stream = CommonTokenStream(lexer)
        parser = OCLsParser(stream)
        tree = parser.oclFile()
        assert parser.getNumberOfSyntaxErrors() == 0
    def test_def(self):
        ocl = "context s def  :	getBurningTransactions() : Set(Transaction) = self.deliveredServices.transactions->iterate(t : Transaction; resultSet : Set( Transaction) = Set{ } | if t.oclIsTypeOf(Burning) then resultSet->including(t) else resultSet endif)"
        input_stream = InputStream(ocl)
        lexer = OCLsLexer(input_stream)
        stream = CommonTokenStream(lexer)
        parser = OCLsParser(stream)
        tree = parser.oclFile()
        assert parser.getNumberOfSyntaxErrors() == 0
    def test_collect_4(self):
        ocl = "context s inv invariant_totalPoints :	(self.deliveredServices->collect( i_Service : Service | i_Service.transactions )->collect( i_Transaction : Transaction | i_Transaction.points )->sum()) < 10000"
        input_stream = InputStream(ocl)
        lexer = OCLsLexer(input_stream)
        stream = CommonTokenStream(lexer)
        parser = OCLsParser(stream)
        tree = parser.oclFile()
        assert parser.getNumberOfSyntaxErrors() == 0
    def test_collect_5(self):
        ocl = "context s inv invariant_ProgramPartner1 :	self.programs->collect( i_LoyaltyProgram : LoyaltyProgram | i_LoyaltyProgram.partners )->select( p : ProgramPartner | p <> self )->isEmpty()"
        input_stream = InputStream(ocl)
        lexer = OCLsLexer(input_stream)
        stream = CommonTokenStream(lexer)
        parser = OCLsParser(stream)
        tree = parser.oclFile()
        assert parser.getNumberOfSyntaxErrors() == 0
    def test_collect_6(self):
        ocl = "context s inv invariant_nrOfParticipants2 :	self.numberOfCustomers = self.programs->collect( i_LoyaltyProgram : LoyaltyProgram | i_LoyaltyProgram.participants )->asSet()->size()"
        input_stream = InputStream(ocl)
        lexer = OCLsLexer(input_stream)
        stream = CommonTokenStream(lexer)
        parser = OCLsParser(stream)
        tree = parser.oclFile()
        assert parser.getNumberOfSyntaxErrors() == 0
    def test_ocl_is_type_of_1(self):
        ocl = "context Burning inv invariant_Burning5 : self.oclIsTypeOf(Earning) = false"
        input_stream = InputStream(ocl)
        lexer = OCLsLexer(input_stream)
        stream = CommonTokenStream(lexer)
        parser = OCLsParser(stream)
        tree = parser.oclFile()
        assert parser.getNumberOfSyntaxErrors() == 0
    def test_ocl_is_type_of_2(self):
        ocl = "context s inv invariant_Burning6 :	self.oclIsKindOf(Earning) = false"
        input_stream = InputStream(ocl)
        lexer = OCLsLexer(input_stream)
        stream = CommonTokenStream(lexer)
        parser = OCLsParser(stream)
        tree = parser.oclFile()
        assert parser.getNumberOfSyntaxErrors() == 0
    def test_ocl_is_kind_of_2(self):
        ocl = "context s inv invariant_Burning4 :	self.oclIsKindOf(Burning) = true"
        input_stream = InputStream(ocl)
        lexer = OCLsLexer(input_stream)
        stream = CommonTokenStream(lexer)
        parser = OCLsParser(stream)
        tree = parser.oclFile()
        assert parser.getNumberOfSyntaxErrors() == 0
    def test_invariant_Burning3(self):
        ocl = "context s inv invariant_Burning3 :	self.oclIsTypeOf(Burning) = true"
        input_stream = InputStream(ocl)
        lexer = OCLsLexer(input_stream)
        stream = CommonTokenStream(lexer)
        parser = OCLsParser(stream)
        tree = parser.oclFile()
        assert parser.getNumberOfSyntaxErrors() == 0
    def test_invariant_Burning2(self):
        ocl = "context s inv invariant_Burning2 :	self.oclIsTypeOf(Transaction) = false"
        input_stream = InputStream(ocl)
        lexer = OCLsLexer(input_stream)
        stream = CommonTokenStream(lexer)
        parser = OCLsParser(stream)
        tree = parser.oclFile()
        assert parser.getNumberOfSyntaxErrors() == 0
    def test_invariant_Burning1(self):
        ocl = "context s inv  invariant_Burning1 :	self.oclIsKindOf(Transaction) = true"
        input_stream = InputStream(ocl)
        lexer = OCLsLexer(input_stream)
        stream = CommonTokenStream(lexer)
        parser = OCLsParser(stream)
        tree = parser.oclFile()
        assert parser.getNumberOfSyntaxErrors() == 0
    def test_collect_7(self):
        ocl = "context TransactionReport inv invariant_dates :	self.lines->collect( i_TransactionReportLine : TransactionReportLine | i_TransactionReportLine.date )->forAll( d : Date | d.isBefore(self.until) and d.isAfter(self.from) )"
        input_stream = InputStream(ocl)
        lexer = OCLsLexer(input_stream)
        stream = CommonTokenStream(lexer)
        parser = OCLsParser(stream)
        tree = parser.oclFile()
        assert parser.getNumberOfSyntaxErrors() == 0
    def test_includes_all(self):
        ocl = "context s inv invariant_cycle :	self.card.transactions->includesAll(self.lines->collect( i_TransactionReportLine : TransactionReportLine | i_TransactionReportLine.transaction ))"
        input_stream = InputStream(ocl)
        lexer = OCLsLexer(input_stream)
        stream = CommonTokenStream(lexer)
        parser = OCLsParser(stream)
        tree = parser.oclFile()
        assert parser.getNumberOfSyntaxErrors() == 0
    def test_derive_int(self):
        ocl = "context TransactionReport::balance : Integer derive : self.card.Membership.account.points"
        input_stream = InputStream(ocl)
        lexer = OCLsLexer(input_stream)
        stream = CommonTokenStream(lexer)
        parser = OCLsParser(stream)
        tree = parser.oclFile()
        assert parser.getNumberOfSyntaxErrors() == 0
    def test_total_earned(self):
        ocl = "context TransactionReport::totalEarned : Integer derive :self.lines->collect( i_TransactionReportLine : TransactionReportLine | i_TransactionReportLine.transaction )->select( i_Transaction : Transaction | i_Transaction.oclIsTypeOf(Earning) )->collect( i_Transaction : Transaction | i_Transaction.points )->sum()"
        input_stream = InputStream(ocl)
        lexer = OCLsLexer(input_stream)
        stream = CommonTokenStream(lexer)
        parser = OCLsParser(stream)
        tree = parser.oclFile()
        assert parser.getNumberOfSyntaxErrors() == 0
    def test_total_burned(self):
        ocl = "context TransactionReport::totalBurned : Integer derive :	self.lines->collect( i_TransactionReportLine : TransactionReportLine | i_TransactionReportLine.transaction )->select( i_Transaction : Transaction | i_Transaction.oclIsTypeOf(Burning) )->collect( i_Transaction : Transaction | i_Transaction.points )->sum()"
        input_stream = InputStream(ocl)
        lexer = OCLsLexer(input_stream)
        stream = CommonTokenStream(lexer)
        parser = OCLsParser(stream)
        tree = parser.oclFile()
        assert parser.getNumberOfSyntaxErrors() == 0
    def test_derive_number(self):
        ocl = "context TransactionReport::number : Integer derive :	self.card.Membership.account.number"
        input_stream = InputStream(ocl)
        lexer = OCLsLexer(input_stream)
        stream = CommonTokenStream(lexer)
        parser = OCLsParser(stream)
        tree = parser.oclFile()
        assert parser.getNumberOfSyntaxErrors() == 0
    def test_derive_name_string(self):
        ocl = "context TransactionReport::name : String derive :	self.card.owner.name"
        input_stream = InputStream(ocl)
        lexer = OCLsLexer(input_stream)
        stream = CommonTokenStream(lexer)
        parser = OCLsParser(stream)
        tree = parser.oclFile()
        assert parser.getNumberOfSyntaxErrors() == 0
    def test_def_1(self):
        ocl = "context CustomerCard def : getTotalPoints(d : Date) : Integer = self.transactions->select( i_Transaction : Transaction | i_Transaction.date.isAfter(d) )->collect( i_Transaction : Transaction | i_Transaction.points )->sum()"
        input_stream = InputStream(ocl)
        lexer = OCLsLexer(input_stream)
        stream = CommonTokenStream(lexer)
        parser = OCLsParser(stream)
        tree = parser.oclFile()
        assert parser.getNumberOfSyntaxErrors() == 0
    def test_inv_cc4(self):
        ocl = "context ccv4 inv invariant_CustomerCard4 :	self.transactions->select( i_Transaction : Transaction | i_Transaction.points > 100 )->notEmpty()"
        input_stream = InputStream(ocl)
        lexer = OCLsLexer(input_stream)
        stream = CommonTokenStream(lexer)
        parser = OCLsParser(stream)
        tree = parser.oclFile()
        assert parser.getNumberOfSyntaxErrors() == 0
    def test_inv_age(self):
        ocl = "context s inv invariant_ofAge :	self.owner.age >= 18"
        input_stream = InputStream(ocl)
        lexer = OCLsLexer(input_stream)
        stream = CommonTokenStream(lexer)
        parser = OCLsParser(stream)
        tree = parser.oclFile()
        assert parser.getNumberOfSyntaxErrors() == 0
    def test_1(self):
        ocl = "context temp inv invariant_CustomerCard3 : self.owner.programs->size() > 0"
        input_stream = InputStream(ocl)
        lexer = OCLsLexer(input_stream)
        stream = CommonTokenStream(lexer)
        parser = OCLsParser(stream)
        tree = parser.oclFile()
        assert parser.getNumberOfSyntaxErrors() == 0
    def test_2(self):
        ocl = "context temp inv invariant_checkDates : self.validFrom.isBefore(self.goodThru)"
        input_stream = InputStream(ocl)
        lexer = OCLsLexer(input_stream)
        stream = CommonTokenStream(lexer)
        parser = OCLsParser(stream)
        tree = parser.oclFile()
        assert parser.getNumberOfSyntaxErrors() == 0
    def test_3(self):
        ocl = "context CustomerCard::valid : Boolean init : true"
        input_stream = InputStream(ocl)
        lexer = OCLsLexer(input_stream)
        stream = CommonTokenStream(lexer)
        parser = OCLsParser(stream)
        tree = parser.oclFile()
        assert parser.getNumberOfSyntaxErrors() == 0
    def test_4(self):
        ocl = "context CustomerCard::printedName : String derive : self.owner.title.concat(' ').concat(self.owner.name)"
        input_stream = InputStream(ocl)
        lexer = OCLsLexer(input_stream)
        stream = CommonTokenStream(lexer)
        parser = OCLsParser(stream)
        tree = parser.oclFile()
        assert parser.getNumberOfSyntaxErrors() == 0
    def test_5(self):
        ocl = "context CustomerCard::myLevel : ServiceLevel derive : self.Membership.currentLevel"
        input_stream = InputStream(ocl)
        lexer = OCLsLexer(input_stream)
        stream = CommonTokenStream(lexer)
        parser = OCLsParser(stream)
        tree = parser.oclFile()
        assert parser.getNumberOfSyntaxErrors() == 0
    def test_6(self):
        ocl = "context CustomerCard::transactions : Set(Transaction) init : Set{}"
        input_stream = InputStream(ocl)
        lexer = OCLsLexer(input_stream)
        stream = CommonTokenStream(lexer)
        parser = OCLsParser(stream)
        tree = parser.oclFile()
        assert parser.getNumberOfSyntaxErrors() == 0
    def test_7(self):
        ocl = "context CustomerCard::getTransactions(until:Date, from:Date) : Set(Transaction) body: self.transactions->select( i_Transaction : Transaction | i_Transaction.date.isAfter(from) and i_Transaction.date.isBefore(until) )"
        input_stream = InputStream(ocl)
        lexer = OCLsLexer(input_stream)
        stream = CommonTokenStream(lexer)
        parser = OCLsParser(stream)
        tree = parser.oclFile()
        assert parser.getNumberOfSyntaxErrors() == 0
    def test_8(self):
        ocl = "context Membership def : getCurrentLevelName() : String = self.currentLevel.name"
        input_stream = InputStream(ocl)
        lexer = OCLsLexer(input_stream)
        stream = CommonTokenStream(lexer)
        parser = OCLsParser(stream)
        tree = parser.oclFile()
        assert parser.getNumberOfSyntaxErrors() == 0
    def test_9(self):
        ocl = "context temp inv invariant_Membership1 : (self.account.points >= 0) or self.account->asSet()->isEmpty()"
        input_stream = InputStream(ocl)
        lexer = OCLsLexer(input_stream)
        stream = CommonTokenStream(lexer)
        parser = OCLsParser(stream)
        tree = parser.oclFile()
        assert parser.getNumberOfSyntaxErrors() == 0
    def test_10(self):
        ocl = "context temp inv invariant_Membership2 : self.participants.cards->collect( i_CustomerCard : CustomerCard | i_CustomerCard.Membership )->includes(self)"
        # ocl = "context temp inv invariant_Membership2 : self.participants.cards->includes(self)"
        input_stream = InputStream(ocl)
        lexer = OCLsLexer(input_stream)
        stream = CommonTokenStream(lexer)
        parser = OCLsParser(stream)
        tree = parser.oclFile()
        assert parser.getNumberOfSyntaxErrors() == 0
    def test_11(self):
        ocl = "context temp inv invariant_noEarnings : programs.partners.deliveredServices->forAll(pointsEarned = 0) implies account->isEmpty()"
        input_stream = InputStream(ocl)
        lexer = OCLsLexer(input_stream)
        stream = CommonTokenStream(lexer)
        parser = OCLsParser(stream)
        tree = parser.oclFile()
        assert parser.getNumberOfSyntaxErrors() == 0
    def test_12(self):
        ocl = "context temp inv invariant_correctCard : self.participants.cards->includes(self.card)"
        input_stream = InputStream(ocl)
        lexer = OCLsLexer(input_stream)
        stream = CommonTokenStream(lexer)
        parser = OCLsParser(stream)
        tree = parser.oclFile()
        assert parser.getNumberOfSyntaxErrors() == 0
    def test_13(self):
        ocl = "context temp inv invariant_Membership3 : self.programs.levels->includes(self.currentLevel)"
        input_stream = InputStream(ocl)
        lexer = OCLsLexer(input_stream)
        stream = CommonTokenStream(lexer)
        parser = OCLsParser(stream)
        tree = parser.oclFile()
        assert parser.getNumberOfSyntaxErrors() == 0
    def test_14(self):
        ocl = "context temp inv invariant_Membership4 : self.account->asSet()->isEmpty()"
        input_stream = InputStream(ocl)
        lexer = OCLsLexer(input_stream)
        stream = CommonTokenStream(lexer)
        parser = OCLsParser(stream)
        tree = parser.oclFile()
        assert parser.getNumberOfSyntaxErrors() == 0
    def test_15(self):
        ocl = "context temp inv invariant_levelAndColor : ((self.currentLevel.name = 'Silver') implies (self.card.color = RandLColor::silver) and self.currentLevel.name = 'Gold') implies self.card.color = RandLColor::gold"
        # ocl = "context temp inv invariant_levelAndColor : ((self.currentLevel.name = 'Silver') "
        input_stream = InputStream(ocl)
        lexer = OCLsLexer(input_stream)
        stream = CommonTokenStream(lexer)
        parser = OCLsParser(stream)
        tree = parser.oclFile()
        assert parser.getNumberOfSyntaxErrors() == 0
    def test_16(self):
        ocl = "context temp inv invariant_Membership5 : self.programs.levels->includes(self.currentLevel)"
        input_stream = InputStream(ocl)
        lexer = OCLsLexer(input_stream)
        stream = CommonTokenStream(lexer)
        parser = OCLsParser(stream)
        tree = parser.oclFile()
        assert parser.getNumberOfSyntaxErrors() == 0
    def test_17(self):
        ocl = "context Service inv invariant_Service5 : 'Anneke'.toUpperCase() = 'ANNEKE'"
        input_stream = InputStream(ocl)
        lexer = OCLsLexer(input_stream)
        stream = CommonTokenStream(lexer)
        parser = OCLsParser(stream)
        tree = parser.oclFile()
        assert parser.getNumberOfSyntaxErrors() == 0
    def test_18(self):
        ocl = "context temp inv invariant_Service5 : 'Anneke'.toUpperCase() = 'ANNEKE'"
        input_stream = InputStream(ocl)
        lexer = OCLsLexer(input_stream)
        stream = CommonTokenStream(lexer)
        parser = OCLsParser(stream)
        tree = parser.oclFile()
        assert parser.getNumberOfSyntaxErrors() == 0
    def test_19(self):
        ocl = "context temp inv invariant_Service6 : 'Anneke'.toLowerCase() = 'anneke'"
        input_stream = InputStream(ocl)
        lexer = OCLsLexer(input_stream)
        stream = CommonTokenStream(lexer)
        parser = OCLsParser(stream)
        tree = parser.oclFile()
        assert parser.getNumberOfSyntaxErrors() == 0
    def test_20(self):
        ocl = "context temp inv invariant_Service7 : ('Anneke and Jos'.substring(12, 14)) = 'Jos'"
        input_stream = InputStream(ocl)
        lexer = OCLsLexer(input_stream)
        stream = CommonTokenStream(lexer)
        parser = OCLsParser(stream)
        tree = parser.oclFile()
        assert parser.getNumberOfSyntaxErrors() == 0
    def test_21(self):
        ocl = "context temp inv invariant_Service4 : ('Anneke '.concat('and Jos')) = 'Anneke and Jos'"
        input_stream = InputStream(ocl)
        lexer = OCLsLexer(input_stream)
        stream = CommonTokenStream(lexer)
        parser = OCLsParser(stream)
        tree = parser.oclFile()
        assert parser.getNumberOfSyntaxErrors() == 0

    def test_22(self):
        ocl = "context temp inv invariant_Service1 : (self.pointsEarned > 0) implies not (self.pointsBurned = 0)"
        input_stream = InputStream(ocl)
        lexer = OCLsLexer(input_stream)
        stream = CommonTokenStream(lexer)
        parser = OCLsParser(stream)
        tree = parser.oclFile()
        assert parser.getNumberOfSyntaxErrors() == 0
    def test_23(self):
        ocl = "context temp inv invariant_Service3 : ('Anneke' = 'Jos') = false"
        input_stream = InputStream(ocl)
        lexer = OCLsLexer(input_stream)
        stream = CommonTokenStream(lexer)
        parser = OCLsParser(stream)
        tree = parser.oclFile()
        assert parser.getNumberOfSyntaxErrors() == 0
    def test_24(self):
        ocl = "context temp inv invariant_Service2 : 'Anneke'.size() = 6"
        input_stream = InputStream(ocl)
        lexer = OCLsLexer(input_stream)
        stream = CommonTokenStream(lexer)
        parser = OCLsParser(stream)
        tree = parser.oclFile()
        assert parser.getNumberOfSyntaxErrors() == 0
    def test_25(self):
        ocl = "context Service::upgradePointsEarned(amount:Integer) : post: self.calcPoints() = self.calcPoints() + amount"
        input_stream = InputStream(ocl)
        lexer = OCLsLexer(input_stream)
        stream = CommonTokenStream(lexer)
        parser = OCLsParser(stream)
        tree = parser.oclFile()
        assert parser.getNumberOfSyntaxErrors() == 0
    def test_26(self):
        ocl = "context Customer inv invariant_Customer4 : self.name = 'Edward'"
        input_stream = InputStream(ocl)
        lexer = OCLsLexer(input_stream)
        stream = CommonTokenStream(lexer)
        parser = OCLsParser(stream)
        tree = parser.oclFile()
        assert parser.getNumberOfSyntaxErrors() == 0
    def test_27(self):
        ocl = "context temp inv invariant_Customer4 : self.name = 'Edward'"
        input_stream = InputStream(ocl)
        lexer = OCLsLexer(input_stream)
        stream = CommonTokenStream(lexer)
        parser = OCLsParser(stream)
        tree = parser.oclFile()
        assert parser.getNumberOfSyntaxErrors() == 0
    def test_28(self):
        ocl = "context temp inv invariant_Customer5 : self.title = 'Mr.'"
        input_stream = InputStream(ocl)
        lexer = OCLsLexer(input_stream)
        stream = CommonTokenStream(lexer)
        parser = OCLsParser(stream)
        tree = parser.oclFile()
        assert parser.getNumberOfSyntaxErrors() == 0
    def test_29(self):
        ocl = "context temp inv invariant_Customer10 : self.programs->collect( i_LoyaltyProgram : LoyaltyProgram | i_LoyaltyProgram.partners )->collectNested( i_ProgramPartner : ProgramPartner | i_ProgramPartner.deliveredServices )->isEmpty()"
        input_stream = InputStream(ocl)
        lexer = OCLsLexer(input_stream)
        stream = CommonTokenStream(lexer)
        parser = OCLsParser(stream)
        tree = parser.oclFile()
        assert parser.getNumberOfSyntaxErrors() == 0
    def test_30(self):
        ocl = "context temp inv invariant_Customer2 : self.name = 'Edward'"
        input_stream = InputStream(ocl)
        lexer = OCLsLexer(input_stream)
        stream = CommonTokenStream(lexer)
        parser = OCLsParser(stream)
        tree = parser.oclFile()
        assert parser.getNumberOfSyntaxErrors() == 0
    def test_31(self):
        ocl = "context temp inv invariant_Customer9 : self.memberships->collect( i_Membership : Membership | i_Membership.account )->reject( i_LoyaltyAccount : LoyaltyAccount | not (i_LoyaltyAccount.points > 0) )->isEmpty()"
        input_stream = InputStream(ocl)
        lexer = OCLsLexer(input_stream)
        stream = CommonTokenStream(lexer)
        parser = OCLsParser(stream)
        tree = parser.oclFile()
        assert parser.getNumberOfSyntaxErrors() == 0
    def test_32(self):
        ocl = "context temp inv invariant_myInvariant23 : self.name = 'Edward'"
        input_stream = InputStream(ocl)
        lexer = OCLsLexer(input_stream)
        stream = CommonTokenStream(lexer)
        parser = OCLsParser(stream)
        tree = parser.oclFile()
        assert parser.getNumberOfSyntaxErrors() == 0
    def test_33(self):
        ocl = "context temp inv invariant_Customer1 : (self.cards->select( i_CustomerCard : CustomerCard | i_CustomerCard.valid = true )->size()) > 1"
        input_stream = InputStream(ocl)
        lexer = OCLsLexer(input_stream)
        stream = CommonTokenStream(lexer)
        parser = OCLsParser(stream)
        tree = parser.oclFile()
        assert parser.getNumberOfSyntaxErrors() == 0
    def test_34(self):
        ocl = "context temp inv invariant_Customer7 : (self.gender = Gender::male) implies self.title = 'Mr.'"
        input_stream = InputStream(ocl)
        lexer = OCLsLexer(input_stream)
        stream = CommonTokenStream(lexer)
        parser = OCLsParser(stream)
        tree = parser.oclFile()
        assert parser.getNumberOfSyntaxErrors() == 0
    def test_35(self):
        ocl = "context temp inv invariant_Customer11 : Set{1, 2, 3 }->iterate(i : Integer; sum : Integer = 0 | sum + i) = 0"
        input_stream = InputStream(ocl)
        lexer = OCLsLexer(input_stream)
        stream = CommonTokenStream(lexer)
        parser = OCLsParser(stream)
        tree = parser.oclFile()
        assert parser.getNumberOfSyntaxErrors() == 0
    def test_36(self):
        ocl = "context temp inv invariant_ANY : self.memberships->collect( i_Membership : Membership | i_Membership.account )->any( i_LoyaltyAccount : LoyaltyAccount | i_LoyaltyAccount.number < 10000 )->asSet()->isEmpty()"
        input_stream = InputStream(ocl)
        lexer = OCLsLexer(input_stream)
        stream = CommonTokenStream(lexer)
        parser = OCLsParser(stream)
        tree = parser.oclFile()
        assert parser.getNumberOfSyntaxErrors() == 0
    def test_37(self):
        ocl = "context temp inv invariant_ofAge : self.age >= 18"
        input_stream = InputStream(ocl)
        lexer = OCLsLexer(input_stream)
        stream = CommonTokenStream(lexer)
        parser = OCLsParser(stream)
        tree = parser.oclFile()
        assert parser.getNumberOfSyntaxErrors() == 0
    def test_38(self):
        ocl = "context temp inv invariant_sizesAgree : self.programs->size() = self.cards->select( i_CustomerCard : CustomerCard | i_CustomerCard.valid = true )->size()"
        input_stream = InputStream(ocl)
        lexer = OCLsLexer(input_stream)
        stream = CommonTokenStream(lexer)
        parser = OCLsParser(stream)
        tree = parser.oclFile()
        assert parser.getNumberOfSyntaxErrors() == 0
    def test_39(self):
        ocl = "context temp inv invariant_Customer8 : self.memberships->collect( i_Membership : Membership | i_Membership.account )->select( i_LoyaltyAccount : LoyaltyAccount | i_LoyaltyAccount.points > 0 )->isEmpty()"
        input_stream = InputStream(ocl)
        lexer = OCLsLexer(input_stream)
        stream = CommonTokenStream(lexer)
        parser = OCLsParser(stream)
        tree = parser.oclFile()
        assert parser.getNumberOfSyntaxErrors() == 0
    def test_40(self):
        ocl = "context temp inv invariant_Customer6 : (self.name = 'Edward') and self.title = 'Mr.' def : wellUsedCards : Set(CustomerCard) = self.cards->select( i_CustomerCard : CustomerCard | (i_CustomerCard.transactions->collect( i_Transaction : Transaction | i_Transaction.points )->sum()) > 10000 )"
        input_stream = InputStream(ocl)
        lexer = OCLsLexer(input_stream)
        stream = CommonTokenStream(lexer)
        parser = OCLsParser(stream)
        tree = parser.oclFile()
        assert parser.getNumberOfSyntaxErrors() == 0
    def test_41(self):
        ocl = "context temp inv invariant_Customer3 : self.name = 'Edward' def : initial : String = self.name.substring(1, 1)"
        input_stream = InputStream(ocl)
        lexer = OCLsLexer(input_stream)
        stream = CommonTokenStream(lexer)
        parser = OCLsParser(stream)
        tree = parser.oclFile()
        assert parser.getNumberOfSyntaxErrors() == 0
    def test_42(self):
        ocl = "context temp inv invariant_Customer12 : self.programs->size() = self.cards->select( i_CustomerCard : CustomerCard | i_CustomerCard.valid = true )->size() def : cardsForProgram(p : LoyaltyProgram) : Sequence(CustomerCard) = p.memberships->collect( i_Membership : Membership | i_Membership.card ) def : loyalToCompanies : Bag(ProgramPartner) = self.programs->collect( i_LoyaltyProgram : LoyaltyProgram | i_LoyaltyProgram.partners )"
        input_stream = InputStream(ocl)
        lexer = OCLsLexer(input_stream)
        stream = CommonTokenStream(lexer)
        parser = OCLsParser(stream)
        tree = parser.oclFile()
        assert parser.getNumberOfSyntaxErrors() == 0
    def test_43(self):
        ocl = "context Customer::birthdayHappens() : post: self.age = self.age + 1"
        input_stream = InputStream(ocl)
        lexer = OCLsLexer(input_stream)
        stream = CommonTokenStream(lexer)
        parser = OCLsParser(stream)
        tree = parser.oclFile()
        assert parser.getNumberOfSyntaxErrors() == 0
    def test_44(self):
        ocl = "context TransactionReportLine::partnerName : String derive : self.transaction.generatedBy.partner.name"
        input_stream = InputStream(ocl)
        lexer = OCLsLexer(input_stream)
        stream = CommonTokenStream(lexer)
        parser = OCLsParser(stream)
        tree = parser.oclFile()
        assert parser.getNumberOfSyntaxErrors() == 0
    def test_45(self):
        ocl = "context TransactionReportLine::serviceDesc : String derive : self.transaction.generatedBy.description"
        input_stream = InputStream(ocl)
        lexer = OCLsLexer(input_stream)
        stream = CommonTokenStream(lexer)
        parser = OCLsParser(stream)
        tree = parser.oclFile()
        assert parser.getNumberOfSyntaxErrors() == 0
    def test_46(self):
        ocl = "context TransactionReportLine::points : Integer derive : self.transaction.points"
        input_stream = InputStream(ocl)
        lexer = OCLsLexer(input_stream)
        stream = CommonTokenStream(lexer)
        parser = OCLsParser(stream)
        tree = parser.oclFile()
        assert parser.getNumberOfSyntaxErrors() == 0
    def test_47(self):
        ocl = "context TransactionReportLine::amount : Real derive : self.transaction.amount"
        input_stream = InputStream(ocl)
        lexer = OCLsLexer(input_stream)
        stream = CommonTokenStream(lexer)
        parser = OCLsParser(stream)
        tree = parser.oclFile()
        assert parser.getNumberOfSyntaxErrors() == 0
    def test_48(self):
        ocl = "context TransactionReportLine::date : Date derive : self.transaction.date"
        input_stream = InputStream(ocl)
        lexer = OCLsLexer(input_stream)
        stream = CommonTokenStream(lexer)
        parser = OCLsParser(stream)
        tree = parser.oclFile()
        assert parser.getNumberOfSyntaxErrors() == 0
    def test_49(self):
        ocl = "context LoyaltyProgram inv invariant_LoyaltyProgram18 : self.participants->forAll( c1 : Customer | self.participants->forAll( c2 : Customer | (c1 <> c2) implies c1.name <> c2.name ) )"
        input_stream = InputStream(ocl)
        lexer = OCLsLexer(input_stream)
        stream = CommonTokenStream(lexer)
        parser = OCLsParser(stream)
        tree = parser.oclFile()
        assert parser.getNumberOfSyntaxErrors() == 0
    def test_50(self):
        ocl = "context temp inv invariant_LoyaltyProgram18 : self.participants->forAll( c1 : Customer | self.participants->forAll( c2 : Customer | (c1 <> c2) implies c1.name <> c2.name ) )"
        input_stream = InputStream(ocl)
        lexer = OCLsLexer(input_stream)
        stream = CommonTokenStream(lexer)
        parser = OCLsParser(stream)
        tree = parser.oclFile()
        assert parser.getNumberOfSyntaxErrors() == 0
    def test_51(self):
        ocl = "context temp inv invariant_LoyaltyProgram1 : self.levels->includesAll(self.memberships->collect( i_Membership : Membership | i_Membership.currentLevel ))"
        input_stream = InputStream(ocl)
        lexer = OCLsLexer(input_stream)
        stream = CommonTokenStream(lexer)
        parser = OCLsParser(stream)
        tree = parser.oclFile()
        assert parser.getNumberOfSyntaxErrors() == 0
    def test_52(self):
        ocl = "context temp inv invariant_LoyaltyProgram17 : self.participants->forAll( c1 : Customer, c2 : Customer | (c1 <> c2) implies c1.name <> c2.name )"
        input_stream = InputStream(ocl)
        lexer = OCLsLexer(input_stream)
        stream = CommonTokenStream(lexer)
        parser = OCLsParser(stream)
        tree = parser.oclFile()
        assert parser.getNumberOfSyntaxErrors() == 0
    def test_53(self):
        ocl = "context temp inv invariant_LoyaltyProgram14 : self.memberships->collect( i_Membership : Membership | i_Membership.account )->isUnique( acc : LoyaltyAccount | acc.number ) def : sortedAccounts : Sequence(LoyaltyAccount) = self.memberships->collect( i_Membership : Membership | i_Membership.account )->sortedBy( i_LoyaltyAccount : LoyaltyAccount | i_LoyaltyAccount.number )"
        input_stream = InputStream(ocl)
        lexer = OCLsLexer(input_stream)
        stream = CommonTokenStream(lexer)
        parser = OCLsParser(stream)
        tree = parser.oclFile()
        assert parser.getNumberOfSyntaxErrors() == 0
    def test_54(self):
        ocl = "context temp inv invariant_LoyaltyProgram10 : Sequence{1 .. 10}->isEmpty()"
        input_stream = InputStream(ocl)
        lexer = OCLsLexer(input_stream)
        stream = CommonTokenStream(lexer)
        parser = OCLsParser(stream)
        tree = parser.oclFile()
        assert parser.getNumberOfSyntaxErrors() == 0
    def test_55(self):
        ocl = "context temp inv invariant_firstLevel : self.levels->first().name = 'Silver'"
        input_stream = InputStream(ocl)
        lexer = OCLsLexer(input_stream)
        stream = CommonTokenStream(lexer)
        parser = OCLsParser(stream)
        tree = parser.oclFile()
        assert parser.getNumberOfSyntaxErrors() == 0
    def test_56(self):
        ocl = "context temp inv invariant_LoyaltyProgram8 : Bag{1, 3, 4, 3, 5}->isEmpty()"
        input_stream = InputStream(ocl)
        lexer = OCLsLexer(input_stream)
        stream = CommonTokenStream(lexer)
        parser = OCLsParser(stream)
        tree = parser.oclFile()
        assert parser.getNumberOfSyntaxErrors() == 0
    def test_57(self):
        # ocl = "context temp inv invariant_knownServiceLevel : self.levels->includesAll(self.memberships->collect( i_Membership : Membership | i_Membership.currentLevel )) def : getServicesByLevel(levelName : String) : Set(Service) = self.levels->select( i_ServiceLevel : ServiceLevel | i_ServiceLevel.name = levelName )->collect( i_ServiceLevel : ServiceLevel | i_ServiceLevel.availableServices )->asSet()"
        ocl = "context temp inv invariant_knownServiceLevel : self.levels->includesAll(self.memberships->collect( i_Membership : Membership | i_Membership.currentLevel )) def : getServicesByLevel(levelName : String) : Set(Service) = self.levels->select( i_ServiceLevel : ServiceLevel | i_ServiceLevel.name = levelName )->collect( i_ServiceLevel : ServiceLevel | i_ServiceLevel.availableServices )->asSet()"
        input_stream = InputStream(ocl)
        lexer = OCLsLexer(input_stream)
        stream = CommonTokenStream(lexer)
        parser = OCLsParser(stream)
        tree = parser.oclFile()
        assert parser.getNumberOfSyntaxErrors() == 0
    def test_58(self):
        ocl = "context temp inv invariant_LoyaltyProgram13 : self.memberships->collect( i_Membership : Membership | i_Membership.account )->isUnique( acc : LoyaltyAccount | acc.number )"
        input_stream = InputStream(ocl)
        lexer = OCLsLexer(input_stream)
        stream = CommonTokenStream(lexer)
        parser = OCLsParser(stream)
        tree = parser.oclFile()
        assert parser.getNumberOfSyntaxErrors() == 0
    def test_59(self):
        ocl = "context temp inv invariant_minServices : (self.partners->collect( i_ProgramPartner : ProgramPartner | i_ProgramPartner.deliveredServices )->size()) >= 1"
        input_stream = InputStream(ocl)
        lexer = OCLsLexer(input_stream)
        stream = CommonTokenStream(lexer)
        parser = OCLsParser(stream)
        tree = parser.oclFile()
        assert parser.getNumberOfSyntaxErrors() == 0
    def test_60(self):
        ocl = "context temp inv invariant_LoyaltyProgram19 : self.memberships->collect( i_Membership : Membership | i_Membership.account )->one( i_LoyaltyAccount : LoyaltyAccount | i_LoyaltyAccount.number < 10000 )"
        input_stream = InputStream(ocl)
        lexer = OCLsLexer(input_stream)
        stream = CommonTokenStream(lexer)
        parser = OCLsParser(stream)
        tree = parser.oclFile()
        assert parser.getNumberOfSyntaxErrors() == 0
    def test_61(self):
        ocl = "context temp inv invariant_LoyaltyProgram12 : self.participants->size() < 10000"
        input_stream = InputStream(ocl)
        lexer = OCLsLexer(input_stream)
        stream = CommonTokenStream(lexer)
        parser = OCLsParser(stream)
        tree = parser.oclFile()
        assert parser.getNumberOfSyntaxErrors() == 0
    def test_62(self):
        ocl = "context temp inv invariant_LoyaltyProgram7 : Sequence{'ape', 'nut'}->isEmpty()"
        input_stream = InputStream(ocl)
        lexer = OCLsLexer(input_stream)
        stream = CommonTokenStream(lexer)
        parser = OCLsParser(stream)
        tree = parser.oclFile()
        assert parser.getNumberOfSyntaxErrors() == 0
    def test_63(self):
        ocl = "context temp inv invariant_LoyaltyProgram11 : Sequence{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}->isEmpty() def : isSaving : Boolean = self.partners->collect( i_ProgramPartner : ProgramPartner | i_ProgramPartner.deliveredServices )->forAll( i_Service : Service | i_Service.pointsEarned = 0 )"
        input_stream = InputStream(ocl)
        lexer = OCLsLexer(input_stream)
        stream = CommonTokenStream(lexer)
        parser = OCLsParser(stream)
        tree = parser.oclFile()
        assert parser.getNumberOfSyntaxErrors() == 0
    def test_64(self):
        ocl = "context temp inv invariant_LoyaltyProgram3 : Set{1, 2, 5, 88}->isEmpty()"
        input_stream = InputStream(ocl)
        lexer = OCLsLexer(input_stream)
        stream = CommonTokenStream(lexer)
        parser = OCLsParser(stream)
        tree = parser.oclFile()
        assert parser.getNumberOfSyntaxErrors() == 0
    def test_65(self):
        ocl = "context temp inv invariant_LoyaltyProgram2 : self.levels->exists( i_ServiceLevel : ServiceLevel | i_ServiceLevel.name = 'basic' )"
        input_stream = InputStream(ocl)
        lexer = OCLsLexer(input_stream)
        stream = CommonTokenStream(lexer)
        parser = OCLsParser(stream)
        tree = parser.oclFile()
        assert parser.getNumberOfSyntaxErrors() == 0
    def test_66(self):
        ocl = "context temp inv invariant_noAccounts : (self.partners->collect( i_ProgramPartner : ProgramPartner | i_ProgramPartner.deliveredServices )->forAll( i_Service : Service | (i_Service.pointsEarned = 0) and i_Service.pointsBurned = 0 )) implies self.memberships->collect( i_Membership : Membership | i_Membership.account )->isEmpty()"
        input_stream = InputStream(ocl)
        lexer = OCLsLexer(input_stream)
        stream = CommonTokenStream(lexer)
        parser = OCLsParser(stream)
        tree = parser.oclFile()
        assert parser.getNumberOfSyntaxErrors() == 0
    def test_67(self):
        ocl = "context temp inv invariant_LoyaltyProgram15 : self.memberships->collect( i_Membership : Membership | i_Membership.account )->isUnique( i_LoyaltyAccount : LoyaltyAccount | i_LoyaltyAccount.number )"
        input_stream = InputStream(ocl)
        lexer = OCLsLexer(input_stream)
        stream = CommonTokenStream(lexer)
        parser = OCLsParser(stream)
        tree = parser.oclFile()
        assert parser.getNumberOfSyntaxErrors() == 0
    def test_68(self):
        ocl = "context temp inv invariant_LoyaltyProgram4 : Set{'apple', 'orange', 'strawberry'}->isEmpty()"
        input_stream = InputStream(ocl)
        lexer = OCLsLexer(input_stream)
        stream = CommonTokenStream(lexer)
        parser = OCLsParser(stream)
        tree = parser.oclFile()
        assert parser.getNumberOfSyntaxErrors() == 0
    def test_69(self):
        ocl = "context temp inv invariant_LoyaltyProgram6 : Sequence{1, 3, 45, 2, 3}->isEmpty()"
        input_stream = InputStream(ocl)
        lexer = OCLsLexer(input_stream)
        stream = CommonTokenStream(lexer)
        parser = OCLsParser(stream)
        tree = parser.oclFile()
        assert parser.getNumberOfSyntaxErrors() == 0
    def test_70(self):
        ocl = "context temp inv invariant_LoyaltyProgram9 : Sequence{1 .. 6 + 4}->isEmpty()"
        input_stream = InputStream(ocl)
        lexer = OCLsLexer(input_stream)
        stream = CommonTokenStream(lexer)
        parser = OCLsParser(stream)
        tree = parser.oclFile()
        assert parser.getNumberOfSyntaxErrors() == 0
    def test_71(self):
        ocl = "context temp inv invariant_LoyaltyProgram5 : OrderedSet{'apple', 'orange', 'strawberry', 'pear'}->isEmpty()"
        input_stream = InputStream(ocl)
        lexer = OCLsLexer(input_stream)
        stream = CommonTokenStream(lexer)
        parser = OCLsParser(stream)
        tree = parser.oclFile()
        assert parser.getNumberOfSyntaxErrors() == 0
    def test_72(self):
        ocl = "context temp inv invariant_LoyaltyProgram16 : self.participants->forAll( i_Customer : Customer | i_Customer.age() <= 70 )"
        input_stream = InputStream(ocl)
        lexer = OCLsLexer(input_stream)
        stream = CommonTokenStream(lexer)
        parser = OCLsParser(stream)
        tree = parser.oclFile()
        assert parser.getNumberOfSyntaxErrors() == 0
    def test_73(self):
        ocl = "context LoyaltyProgram::selectPopularPartners(d:Date) : Set(ProgramPartner) post: let popularTrans : Set(Transaction) = result->collect( i_ProgramPartner : ProgramPartner | i_ProgramPartner.deliveredServices )->collect( i_Service : Service | i_Service.transactions )->asSet() in (popularTrans->forAll( i_Transaction : Transaction | i_Transaction.date.isAfter(d) )) and (popularTrans->select( i_Transaction : Transaction | i_Transaction.amount > 500.00 )->size()) > 20000"
        input_stream = InputStream(ocl)
        lexer = OCLsLexer(input_stream)
        stream = CommonTokenStream(lexer)
        parser = OCLsParser(stream)
        tree = parser.oclFile()
        assert parser.getNumberOfSyntaxErrors() == 0
    def test_74(self):
        ocl = "context LoyaltyProgram::addService(s:Service, l:ServiceLevel, p:ProgramPartner) : pre levelsIncludesArgL: self.levels->includes(l) post servicesIncludesArgS: self.levels->collect( i_ServiceLevel : ServiceLevel | i_ServiceLevel.availableServices )->includes(s) pre partnersIncludesP: self.partners->includes(p) post servicesIncludesP: self.partners->collect( i_ProgramPartner : ProgramPartner | i_ProgramPartner.deliveredServices )->includes(s)"
        input_stream = InputStream(ocl)
        lexer = OCLsLexer(input_stream)
        stream = CommonTokenStream(lexer)
        parser = OCLsParser(stream)
        tree = parser.oclFile()
        assert parser.getNumberOfSyntaxErrors() == 0
    def test_75(self):
        ocl = "context LoyaltyProgram::getServices(pp:ProgramPartner) : Set(Service) body: if self.partners->includes(pp) then pp.deliveredServices else Set{} endif"
        input_stream = InputStream(ocl)
        lexer = OCLsLexer(input_stream)
        stream = CommonTokenStream(lexer)
        parser = OCLsParser(stream)
        tree = parser.oclFile()
        assert parser.getNumberOfSyntaxErrors() == 0
    def test_76(self):
        ocl = "context LoyaltyProgram::enrollAndCreateCustomer(n : String, d: Date ) : Customer post: ((result.oclIsNew() and result.name = n) and result.dateOfBirth = d) and self.participants->includes(result) pre constantlyTrue: true"
        input_stream = InputStream(ocl)
        lexer = OCLsLexer(input_stream)
        stream = CommonTokenStream(lexer)
        parser = OCLsParser(stream)
        tree = parser.oclFile()
        assert parser.getNumberOfSyntaxErrors() == 0

    def test_77(self):
        ocl = "context LoyaltyProgram::enroll(c:Customer) : OclVoid post: self.participants = self.participants->including(c) post: self.participants = self.participants->including(c) pre: c.name <> '' pre: c.name <> '' pre: not self.participants->includes(c) post: self.participants = self.participants->including(c) post: self.participants = self.participants->including(c) pre: not self.participants->includes(c)"
        input_stream = InputStream(ocl)
        lexer = OCLsLexer(input_stream)
        stream = CommonTokenStream(lexer)
        parser = OCLsParser(stream)
        tree = parser.oclFile()
        assert parser.getNumberOfSyntaxErrors() == 0
    def test_78(self):
        ocl = "context meeting inv: self.start < self.end and self.start < 5 and self.end >5"
        input_stream = InputStream(ocl)
        lexer = OCLsLexer(input_stream)
        stream = CommonTokenStream(lexer)
        parser = OCLsParser(stream)
        tree = parser.oclFile()
        # listener = OCLsListener()
        # walker = ParseTreeWalker()
        # walker.walk(listener,tree)
        assert parser.getNumberOfSyntaxErrors() == 0


    def test_79(self):
        ocl = "context meeting inv: self.start < 10"
        input_stream = InputStream(ocl)
        lexer = OCLsLexer(input_stream)
        stream = CommonTokenStream(lexer)
        parser = OCLsParser(stream)
        tree = parser.oclFile()
        assert parser.getNumberOfSyntaxErrors() == 0
    def test_80(self):
        ocl = "context meeting inv:-1 <self.start "
        input_stream = InputStream(ocl)
        lexer = OCLsLexer(input_stream)
        stream = CommonTokenStream(lexer)
        parser = OCLsParser(stream)
        tree = parser.oclFile()
        assert parser.getNumberOfSyntaxErrors() == 0
if __name__ == '__main__':
    # main(sys.argv)
    print("in main")