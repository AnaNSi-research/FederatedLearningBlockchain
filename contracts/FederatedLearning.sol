// SPDX-License-Identifier: MIT

// pragma solidity ^0.6.6;
pragma solidity ^0.6.0;
pragma experimental ABIEncoderV2;

import "@chainlink/contracts/src/v0.6/interfaces/AggregatorV3Interface.sol";
import "@chainlink/contracts/src/v0.6/vendor/SafeMathChainlink.sol";
import "@openzeppelin/contracts/access/Ownable.sol";

contract FederatedLearning is Ownable {
    using SafeMathChainlink for uint256;
    enum FL_STATE {
        CLOSED,
        OPEN,
        START,
        LEARNING
    }
    FL_STATE public fl_state;
    address[] public collaborators;
    bytes public model;
    bytes public compile_info;
    bytes public aggregated_weights;
    bytes[] public weights;

    /*
    uint public timeoutDuration = 570; // Timeout duration in seconds
    uint public timeoutStart; // Timestamp when the timeout starts
    bool public isTimeoutStarted = false; // Flag indicating whether the timeout has started
    */

    mapping(address => mapping(string => bool)) public hasCalledFunction; // only once
    mapping(string => uint) public everyoneHasCalled; // for all

    event ChangeState(string old_state, string new_state);
    event EveryCollaboratorhasCalledOnlyOnce(string functionName);
    event AggregatedWeightsReady();
    // event TimeoutExpired(string functionName);

    // if you're following along with the freecodecamp video
    // Please see https://github.com/PatrickAlphaC/fund_me
    // to get the starting solidity contract code, it'll be slightly different than this!
    constructor() public {
        fl_state = FL_STATE.CLOSED;
    }

    modifier onlyAuthorized() {
        require(isAuthorized(msg.sender), "Unauthorized user");
        _;
    }

    modifier everyCollaboratorHasCalledOnce(string memory functionName) {
        require(!hasCalledFunction[msg.sender][functionName], "This function can only be called only once per collaborator");
        hasCalledFunction[msg.sender][functionName] = true;

        everyoneHasCalled[functionName]++;
        if(everyoneHasCalled[functionName] == collaborators.length) {
            emit EveryCollaboratorhasCalledOnlyOnce(functionName);
        }

        _;
    }

    function isAuthorized(address _user) public view returns (bool) {
        if (owner() == _user) {
            return true;
        }
        for (uint i = 0; i < collaborators.length; i++) {
            if (collaborators[i] == _user) {
                return true;
            }
        }
        return false;
    }

    /*
    function startTimeout() public onlyOwner {
        require(!isTimeoutStarted, "Timeout has already been started");
        timeoutStart = block.timestamp;
        isTimeoutStarted = true;
    }

    function checkTimeout(string memory phase) public onlyOwner {
        require(isTimeoutStarted, "Timeout has not been started yet");
        require(block.timestamp >= timeoutStart + timeoutDuration, "Timeout has not expired yet");
        isTimeoutStarted = false;
        emit TimeoutExpired(phase);
    }

    function resetTimeout() public onlyOwner {
        isTimeoutStarted = false;
    }
    */

    function open() public onlyOwner {
        require(fl_state == FL_STATE.CLOSED);
        fl_state = FL_STATE.OPEN;
        emit ChangeState("CLOSED", get_state());
    }

    function add_collaborator(address _collaborator) public onlyOwner {
        require(fl_state == FL_STATE.OPEN);
        collaborators.push(_collaborator);
    }

    function send_model(bytes memory _model) public onlyOwner {
        require(fl_state == FL_STATE.OPEN);
        model = _model;
    }

    function send_compile_info(bytes memory _compile_info) public onlyOwner {
        require(fl_state == FL_STATE.OPEN);
        compile_info = _compile_info;
    }

    function start() public onlyOwner {
        require(fl_state == FL_STATE.OPEN);
        fl_state = FL_STATE.START;
        emit ChangeState("OPEN", get_state());
    }

    function retrieve_model() public onlyAuthorized everyCollaboratorHasCalledOnce("retrieve_model") returns (bytes memory){
        require(fl_state == FL_STATE.START);
        return model;
    }

    function retrieve_compile_info() public onlyAuthorized everyCollaboratorHasCalledOnce("retrieve_compile_info") returns (bytes memory) {
        require(fl_state == FL_STATE.START);
        return compile_info;
    }

    function learning() public onlyOwner {
        require(fl_state == FL_STATE.START);
        fl_state = FL_STATE.LEARNING;
        emit ChangeState("START", get_state());
    }

    function send_weights(bytes memory _weights) public onlyAuthorized everyCollaboratorHasCalledOnce("send_weights"){
        require(fl_state == FL_STATE.LEARNING);
        require(weights.length <= collaborators.length);
        weights.push(_weights);
    }

    function retrieve_weights() public view onlyOwner returns (bytes[] memory){
        require(fl_state == FL_STATE.LEARNING);
        return weights;
    }

    function reset_send_weights() public onlyOwner {
        for (uint256 i = 0; i < collaborators.length; i++) {
            address collaborator = collaborators[i];
            delete hasCalledFunction[collaborator]["send_weights"];
        }
        delete everyoneHasCalled["send_weights"];
    }

    function send_aggregated_weights(bytes memory _weights) public onlyOwner {
        require(fl_state == FL_STATE.LEARNING);
        aggregated_weights = _weights;
        delete weights;

        emit AggregatedWeightsReady();
    }

    function retrieve_aggregated_weights() public onlyAuthorized everyCollaboratorHasCalledOnce("retrieve_aggregated_weights") returns (bytes memory) {
        require(fl_state == FL_STATE.LEARNING);
        return aggregated_weights;
    }

    function reset_retrieve_aggregated_weights() public onlyOwner {
        for (uint256 i = 0; i < collaborators.length; i++) {
            address collaborator = collaborators[i];
            delete hasCalledFunction[collaborator]["retrieve_aggregated_weights"];
        }
        delete everyoneHasCalled["retrieve_aggregated_weights"];
    }

    function close() public onlyOwner {
        require(fl_state == FL_STATE.LEARNING);
        fl_state = FL_STATE.CLOSED;
        emit ChangeState("LEARNING", get_state());
    }


    function get_state() public view returns (string memory) {
        if (fl_state == FL_STATE.CLOSED) return "CLOSED";
        if (fl_state == FL_STATE.OPEN) return "OPEN";
        if (fl_state == FL_STATE.START) return "START";
        if (fl_state == FL_STATE.LEARNING) return "LEARNING";
        return "No State";
    }


    function get_collaborators() public view returns (address[] memory) {
        return collaborators;
    }

    function get_model() public onlyAuthorized view returns (bytes memory) {
        return model;
    }

    function get_compile_info() public onlyAuthorized view returns (bytes memory) {
        return compile_info;
    }

    function get_aggregated_weights() public onlyAuthorized view returns (bytes memory) {
        return aggregated_weights;
    }

    function get_weights() public view onlyAuthorized returns (bytes[] memory) {
        return weights;
    }
}
