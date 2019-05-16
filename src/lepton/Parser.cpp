/* +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
 * -------------------------------------------------------------------------- *
 *                                   Lepton                                   *
 * -------------------------------------------------------------------------- *
 * This is part of the Lepton expression parser originating from              *
 * Simbios, the NIH National Center for Physics-Based Simulation of           *
 * Biological Structures at Stanford, funded under the NIH Roadmap for        *
 * Medical Research, grant U54 GM072970. See https://simtk.org.               *
 *                                                                            *
 * Portions copyright (c) 2013-2016 Stanford University and the Authors.      *
 * Authors: Peter Eastman                                                     *
 * Contributors:                                                              *
 *                                                                            *
 * Permission is hereby granted, free of charge, to any person obtaining a    *
 * copy of this software and associated documentation files (the "Software"), *
 * to deal in the Software without restriction, including without limitation  *
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,   *
 * and/or sell copies of the Software, and to permit persons to whom the      *
 * Software is furnished to do so, subject to the following conditions:       *
 *                                                                            *
 * The above copyright notice and this permission notice shall be included in *
 * all copies or substantial portions of the Software.                        *
 *                                                                            *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR *
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,   *
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL    *
 * THE AUTHORS, CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,    *
 * DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR      *
 * OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE  *
 * USE OR OTHER DEALINGS IN THE SOFTWARE.                                     *
 * -------------------------------------------------------------------------- *
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ */
/* -------------------------------------------------------------------------- *
 *                                   lepton                                   *
 * -------------------------------------------------------------------------- *
 * This is part of the lepton expression parser originating from              *
 * Simbios, the NIH National Center for Physics-Based Simulation of           *
 * Biological Structures at Stanford, funded under the NIH Roadmap for        *
 * Medical Research, grant U54 GM072970. See https://simtk.org.               *
 *                                                                            *
 * Portions copyright (c) 2009-2015 Stanford University and the Authors.      *
 * Authors: Peter Eastman                                                     *
 * Contributors:                                                              *
 *                                                                            *
 * Permission is hereby granted, free of charge, to any person obtaining a    *
 * copy of this software and associated documentation files (the "Software"), *
 * to deal in the Software without restriction, including without limitation  *
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,   *
 * and/or sell copies of the Software, and to permit persons to whom the      *
 * Software is furnished to do so, subject to the following conditions:       *
 *                                                                            *
 * The above copyright notice and this permission notice shall be included in *
 * all copies or substantial portions of the Software.                        *
 *                                                                            *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR *
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,   *
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL    *
 * THE AUTHORS, CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,    *
 * DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR      *
 * OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE  *
 * USE OR OTHER DEALINGS IN THE SOFTWARE.                                     *
 * -------------------------------------------------------------------------- */

#include "Parser.h"
#include "CustomFunction.h"
#include "Exception.h"
#include "ExpressionTreeNode.h"
#include "Operation.h"
#include "ParsedExpression.h"
#include <cctype>
#include <iostream>

namespace PLMD {
using namespace lepton;
using namespace std;

namespace lepton {

static const string Digits = "0123456789";
static const string Operators = "+-*/^";
static const bool LeftAssociative[] = {true, true, true, true, false};
static const int Precedence[] = {0, 0, 1, 1, 3};
static const Operation::Id OperationId[] = {Operation::ADD, Operation::SUBTRACT, Operation::MULTIPLY, Operation::DIVIDE, Operation::POWER};

const std::map<std::string, double> & Constants() {
  static const std::map<std::string, double> constants = {
  {"e", std::exp(1.0)},
  {"log2e", 1.0/std::log(2.0)},
  {"log10e", 1.0/std::log(10.0)},
  {"ln2", std::log(2.0)},
  {"ln10", std::log(10.0)},
  {"pi", 3.14159265358979323844},
  {"pi_2", 3.14159265358979323844*0.5},
  {"pi_4", 3.14159265358979323844*0.25},
//  {"1_pi", 1.0/pi},
//  {"2_pi", 2.0/pi},
//  {"2_sqrtpi", 2.0/std::sqrt(pi)},
  {"sqrt2", std::sqrt(2.0)},
  {"sqrt1_2", std::sqrt(0.5)}
  };
  return constants;
}

}


class lepton::ParseToken {
public:
    enum Type {Number, Operator, Variable, Function, LeftParen, RightParen, Comma, Whitespace};

    ParseToken(string text, Type type) : text(text), type(type) {
    }
    const string& getText() const {
        return text;
    }
    Type getType() const {
        return type;
    }
private:
    string text;
    Type type;
};

string Parser::trim(const string& expression) {
    // Remove leading and trailing spaces.
    
    int start, end;
    for (start = 0; start < (int) expression.size() && isspace(expression[start]); start++)
        ;
    for (end = (int) expression.size()-1; end > start && isspace(expression[end]); end--)
        ;
    if (start == end && isspace(expression[end]))
        return "";
    return expression.substr(start, end-start+1);
}

ParseToken Parser::getNextToken(const string& expression, int start) {
    char c = expression[start];
    if (c == '(')
        return ParseToken("(", ParseToken::LeftParen);
    if (c == ')')
        return ParseToken(")", ParseToken::RightParen);
    if (c == ',')
        return ParseToken(",", ParseToken::Comma);
    if (Operators.find(c) != string::npos)
        return ParseToken(string(1, c), ParseToken::Operator);
    if (isspace(c)) {
        // White space

        for (int pos = start+1; pos < (int) expression.size(); pos++) {
            if (!isspace(expression[pos]))
                return ParseToken(expression.substr(start, pos-start), ParseToken::Whitespace);
        }
        return ParseToken(expression.substr(start, string::npos), ParseToken::Whitespace);
    }
    if (c == '.' || Digits.find(c) != string::npos) {
        // A number

        bool foundDecimal = (c == '.');
        bool foundExp = false;
        int pos;
        for (pos = start+1; pos < (int) expression.size(); pos++) {
            c = expression[pos];
            if (Digits.find(c) != string::npos)
                continue;
            if (c == '.' && !foundDecimal) {
                foundDecimal = true;
                continue;
            }
            if ((c == 'e' || c == 'E') && !foundExp) {
                foundExp = true;
                if (pos < (int) expression.size()-1 && (expression[pos+1] == '-' || expression[pos+1] == '+'))
                    pos++;
                continue;
            }
            break;
        }
        return ParseToken(expression.substr(start, pos-start), ParseToken::Number);
    }

    // A variable, function, or left parenthesis

    for (int pos = start; pos < (int) expression.size(); pos++) {
        c = expression[pos];
        if (c == '(')
            return ParseToken(expression.substr(start, pos-start+1), ParseToken::Function);
        if (Operators.find(c) != string::npos || c == ',' || c == ')' || isspace(c))
            return ParseToken(expression.substr(start, pos-start), ParseToken::Variable);
    }
    return ParseToken(expression.substr(start, string::npos), ParseToken::Variable);
}

vector<ParseToken> Parser::tokenize(const string& expression) {
    vector<ParseToken> tokens;
    int pos = 0;
    while (pos < (int) expression.size()) {
        ParseToken token = getNextToken(expression, pos);
        if (token.getType() != ParseToken::Whitespace)
            tokens.push_back(token);
        pos += (int) token.getText().size();
    }
    return tokens;
}

ParsedExpression Parser::parse(const string& expression) {
    return parse(expression, map<string, CustomFunction*>());
}

ParsedExpression Parser::parse(const string& expression, const map<string, CustomFunction*>& customFunctions) {
    try {
        // First split the expression into subexpressions.

        string primaryExpression = expression;
        vector<string> subexpressions;
        while (true) {
            string::size_type pos = primaryExpression.find_last_of(';');
            if (pos == string::npos)
                break;
            string sub = trim(primaryExpression.substr(pos+1));
            if (sub.size() > 0)
                subexpressions.push_back(sub);
            primaryExpression = primaryExpression.substr(0, pos);
        }

        // Parse the subexpressions.

        map<string, ExpressionTreeNode> subexpDefs;
        for (int i = 0; i < (int) subexpressions.size(); i++) {
            string::size_type equalsPos = subexpressions[i].find('=');
            if (equalsPos == string::npos)
                throw Exception("subexpression does not specify a name");
            string name = trim(subexpressions[i].substr(0, equalsPos));
            if (name.size() == 0)
                throw Exception("subexpression does not specify a name");
            vector<ParseToken> tokens = tokenize(subexpressions[i].substr(equalsPos+1));
            int pos = 0;
            subexpDefs[name] = parsePrecedence(tokens, pos, customFunctions, subexpDefs, 0);
            if (pos != tokens.size())
                throw Exception("unexpected text at end of subexpression: "+tokens[pos].getText());
        }

        // Now parse the primary expression.

        vector<ParseToken> tokens = tokenize(primaryExpression);
        int pos = 0;
        ExpressionTreeNode result = parsePrecedence(tokens, pos, customFunctions, subexpDefs, 0);
        if (pos != tokens.size())
            throw Exception("unexpected text at end of expression: "+tokens[pos].getText());
        return ParsedExpression(result);
    }
    catch (Exception& ex) {
        throw Exception("Parse error in expression \""+expression+"\": "+ex.what());
    }
}

ExpressionTreeNode Parser::parsePrecedence(const vector<ParseToken>& tokens, int& pos, const map<string, CustomFunction*>& customFunctions,
            const map<string, ExpressionTreeNode>& subexpressionDefs, int precedence) {
    if (pos == tokens.size())
        throw Exception("unexpected end of expression");

    // Parse the next value (number, variable, function, parenthesized expression)

    ParseToken token = tokens[pos];
    ExpressionTreeNode result;
    if (token.getType() == ParseToken::Number) {
        double value;
        stringstream(token.getText()) >> value;
        result = ExpressionTreeNode(new Operation::Constant(value));
        pos++;
    }
    else if (token.getType() == ParseToken::Variable) {
        map<string, ExpressionTreeNode>::const_iterator subexp = subexpressionDefs.find(token.getText());
        if (subexp == subexpressionDefs.end()) {
            Operation* op = new Operation::Variable(token.getText());
            result = ExpressionTreeNode(op);
        }
        else
            result = subexp->second;
        pos++;
    }
    else if (token.getType() == ParseToken::LeftParen) {
        pos++;
        result = parsePrecedence(tokens, pos, customFunctions, subexpressionDefs, 0);
        if (pos == tokens.size() || tokens[pos].getType() != ParseToken::RightParen)
            throw Exception("unbalanced parentheses");
        pos++;
    }
    else if (token.getType() == ParseToken::Function) {
        pos++;
        vector<ExpressionTreeNode> args;
        bool moreArgs;
        do {
            args.push_back(parsePrecedence(tokens, pos, customFunctions, subexpressionDefs, 0));
            moreArgs = (pos < (int) tokens.size() && tokens[pos].getType() == ParseToken::Comma);
            if (moreArgs)
                pos++;
        } while (moreArgs);
        if (pos == tokens.size() || tokens[pos].getType() != ParseToken::RightParen)
            throw Exception("unbalanced parentheses");
        pos++;
        Operation* op = getFunctionOperation(token.getText(), customFunctions);
        try {
            result = ExpressionTreeNode(op, args);
        }
        catch (...) {
            delete op;
            throw;
        }
    }
    else if (token.getType() == ParseToken::Operator && token.getText() == "-") {
        pos++;
        ExpressionTreeNode toNegate = parsePrecedence(tokens, pos, customFunctions, subexpressionDefs, 2);
        result = ExpressionTreeNode(new Operation::Negate(), toNegate);
    }
    else
        throw Exception("unexpected token: "+token.getText());

    // Now deal with the next binary operator.

    while (pos < (int) tokens.size() && tokens[pos].getType() == ParseToken::Operator) {
        token = tokens[pos];
        int opIndex = (int) Operators.find(token.getText());
        int opPrecedence = Precedence[opIndex];
        if (opPrecedence < precedence)
            return result;
        pos++;
        ExpressionTreeNode arg = parsePrecedence(tokens, pos, customFunctions, subexpressionDefs, LeftAssociative[opIndex] ? opPrecedence+1 : opPrecedence);
        Operation* op = getOperatorOperation(token.getText());
        try {
            result = ExpressionTreeNode(op, result, arg);
        }
        catch (...) {
            delete op;
            throw;
        }
    }
    return result;
}

Operation* Parser::getOperatorOperation(const std::string& name) {
    switch (OperationId[Operators.find(name)]) {
        case Operation::ADD:
            return new Operation::Add();
        case Operation::SUBTRACT:
            return new Operation::Subtract();
        case Operation::MULTIPLY:
            return new Operation::Multiply();
        case Operation::DIVIDE:
            return new Operation::Divide();
        case Operation::POWER:
            return new Operation::Power();
        default:
            throw Exception("unknown operator");
    }
}

Operation* Parser::getFunctionOperation(const std::string& name, const map<string, CustomFunction*>& customFunctions) {

    const static map<string, Operation::Id> opMap ={
        { "sqrt" , Operation::SQRT },
        { "exp" , Operation::EXP },
        { "log" , Operation::LOG },
        { "sin" , Operation::SIN },
        { "cos" , Operation::COS },
        { "sec" , Operation::SEC },
        { "csc" , Operation::CSC },
        { "tan" , Operation::TAN },
        { "cot" , Operation::COT },
        { "asin" , Operation::ASIN },
        { "acos" , Operation::ACOS },
        { "atan" , Operation::ATAN },
        { "sinh" , Operation::SINH },
        { "cosh" , Operation::COSH },
        { "tanh" , Operation::TANH },
        { "erf" , Operation::ERF },
        { "erfc" , Operation::ERFC },
        { "step" , Operation::STEP },
        { "delta" , Operation::DELTA },
        { "nandelta" , Operation::NANDELTA },
        { "square" , Operation::SQUARE },
        { "cube", Operation::CUBE },
        { "recip" , Operation::RECIPROCAL },
        { "min" , Operation::MIN },
        { "max" , Operation::MAX },
        { "abs" , Operation::ABS },
        { "floor" , Operation::FLOOR },
        { "ceil" , Operation::CEIL },
        { "select" , Operation::SELECT },
        { "acot" , Operation::ACOT },
        { "asec" , Operation::ASEC },
        { "acsc" , Operation::ACSC },
        { "coth" , Operation::COTH },
        { "sech" , Operation::SECH },
        { "csch" , Operation::CSCH },
        { "asinh" , Operation::ASINH },
        { "acosh" , Operation::ACOSH },
        { "atanh" , Operation::ATANH },
        { "acoth" , Operation::ACOTH },
        { "asech" , Operation::ASECH },
        { "acsch" , Operation::ACSCH },
        { "atan2" , Operation::ATAN2 },
    };
    string trimmed = name.substr(0, name.size()-1);

    // First check custom functions.

    map<string, CustomFunction*>::const_iterator custom = customFunctions.find(trimmed);
    if (custom != customFunctions.end())
        return new Operation::Custom(trimmed, custom->second->clone());

    // Now try standard functions.

    map<string, Operation::Id>::const_iterator iter = opMap.find(trimmed);
    if (iter == opMap.end())
        throw Exception("unknown function: "+trimmed);
    switch (iter->second) {
        case Operation::SQRT:
            return new Operation::Sqrt();
        case Operation::EXP:
            return new Operation::Exp();
        case Operation::LOG:
            return new Operation::Log();
        case Operation::SIN:
            return new Operation::Sin();
        case Operation::COS:
            return new Operation::Cos();
        case Operation::SEC:
            return new Operation::Sec();
        case Operation::CSC:
            return new Operation::Csc();
        case Operation::TAN:
            return new Operation::Tan();
        case Operation::COT:
            return new Operation::Cot();
        case Operation::ASIN:
            return new Operation::Asin();
        case Operation::ACOS:
            return new Operation::Acos();
        case Operation::ATAN:
            return new Operation::Atan();
        case Operation::SINH:
            return new Operation::Sinh();
        case Operation::COSH:
            return new Operation::Cosh();
        case Operation::TANH:
            return new Operation::Tanh();
        case Operation::ERF:
            return new Operation::Erf();
        case Operation::ERFC:
            return new Operation::Erfc();
        case Operation::STEP:
            return new Operation::Step();
        case Operation::DELTA:
            return new Operation::Delta();
        case Operation::NANDELTA:
            return new Operation::Nandelta();
        case Operation::SQUARE:
            return new Operation::Square();
        case Operation::CUBE:
            return new Operation::Cube();
        case Operation::RECIPROCAL:
            return new Operation::Reciprocal();
        case Operation::MIN:
            return new Operation::Min();
        case Operation::MAX:
            return new Operation::Max();
        case Operation::ABS:
            return new Operation::Abs();
        case Operation::FLOOR:
            return new Operation::Floor();
        case Operation::CEIL:
            return new Operation::Ceil();
        case Operation::SELECT:
            return new Operation::Select();
        case Operation::ACOT:
            return new Operation::Acot();
        case Operation::ASEC:
            return new Operation::Asec();
        case Operation::ACSC:
            return new Operation::Acsc();
        case Operation::COTH:
            return new Operation::Coth();
        case Operation::SECH:
            return new Operation::Sech();
        case Operation::CSCH:
            return new Operation::Csch();
        case Operation::ASINH:
            return new Operation::Asinh();
        case Operation::ACOSH:
            return new Operation::Acosh();
        case Operation::ATANH:
            return new Operation::Atanh();
        case Operation::ACOTH:
            return new Operation::Acoth();
        case Operation::ASECH:
            return new Operation::Asech();
        case Operation::ACSCH:
            return new Operation::Acsch();
        case Operation::ATAN2:
            return new Operation::Atan2();
        default:
            throw Exception("unknown function");
    }
}
}
